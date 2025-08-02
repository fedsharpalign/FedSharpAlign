# In[1]:


import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import pickle

from torchvision.models import densenet121, DenseNet121_Weights


from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import json
from torch.utils.data import WeightedRandomSampler
import numpy as np
from collections import Counter
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()


# In[2]:


print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA device found!")


# In[ ]:


with open('../canada_india_bangladesh_spain.pkl', 'rb') as f:
    data_dict = pickle.load(f)

site_names = list(data_dict.keys())
num_clients = len(site_names)
print(f"Loaded {num_clients} clients: {site_names}")


# In[4]:


from PIL import Image
from torch.utils.data import Dataset

class OralCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")


        json_path = img_path.replace(".jpg", ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)


                if "/Bangladesh/" in img_path and "bboxes" in data:
                    is_crop = "_crop" in os.path.basename(img_path)
                    if not is_crop:  
                        bbox = data["bboxes"][0]
                        x, y, w, h = map(int, bbox)
                        if w > 0 and h > 0:
                            img = img.crop((x, y, x + w, y + h))

                elif "shapes" in data:
                    points = data['shapes'][0]['points']
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        img = img.crop(bbox)

            except Exception as e:
                print(f"WFailed to process JSON for {img_path}: {e}")

        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# In[5]:


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_loaders, val_loaders, test_loaders = [], [], []
site_names = list(data_dict.keys())

print(f"Loading data for {len(site_names)} clients: {site_names}")

for site in site_names:
    X_train, y_train = data_dict[site]['train']
    X_val, y_val = data_dict[site]['val']
    X_test, y_test = data_dict[site]['test']

    print(f"[{site}] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    trainset = OralCancerDataset(X_train, y_train, transform=train_transform)
    valset   = OralCancerDataset(X_val, y_val, transform=val_transform)
    testset  = OralCancerDataset(X_test, y_test, transform=val_transform)

    
    class_counts = Counter(y_train)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # Dataloaders
    train_loader = DataLoader(trainset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


    train_loaders.append(train_loader)
    val_loaders.append(val_loader)
    test_loaders.append(test_loader)


# In[6]:


def get_model():
    weights = DenseNet121_Weights.IMAGENET1K_V1  
    model = densenet121(weights=weights)
    
    model.classifier = nn.Linear(model.classifier.in_features, 3)

    return model


# In[ ]:


BETA = 0.5
def cosine_regularization(model, server_model):
    cosine_loss = 0.0
    for (name, param), server_param in zip(model.named_parameters(), server_model.parameters()):
        if not any(k in name.lower() for k in ['bn', 'norm']):
            p = param.view(-1)
            sp = server_param.view(-1).detach()
            cos_sim = F.cosine_similarity(p, sp, dim=0)
            cosine_loss += (1.0 - cos_sim)
    return cosine_loss



def confidence_penalty(logits, beta=1.0):
    scaled_logits = logits
    probs = F.softmax(scaled_logits, dim=1)
    entropy = - (probs * torch.log(probs + 1e-12)).sum(dim=1)
    return beta * entropy.mean()


def local_train(model, loader, loss_fn, optimizer, device, scaler, 
                local_epochs=1, method=None, server_model=None, mu=0.01, use_wp=False, overfit_gap=None):
    
    model.train()
    model.to(device)
    total_loss, correct, total = 0.0, 0, 0

    for epoch in range(local_epochs):

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{local_epochs}",
            ncols=100,
            leave=True,            
            mininterval=0.3,       
            dynamic_ncols=True     
        )

        for data, target in pbar:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_wp:
                # 1st forward + backward on original weights
                output = model(data)
                loss = loss_fn(output, target)

                if overfit_gap is not None and overfit_gap > 0:
                    loss -= overfit_gap * confidence_penalty(output, beta=BETA)

                if method == "fedprox" and server_model is not None:
                    prox_term = 0.0
                    for param, server_param in zip(model.parameters(), server_model.parameters()):
                        prox_term += torch.sum((param - server_param.detach()) ** 2)
                    loss += (mu / 2.0) * prox_term
                    
                elif method == "cosine" and server_model is not None:
                    loss += mu * cosine_regularization(model, server_model)
                    
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # 2nd forward + backward on perturbed weights
                perturbed_output = model(data)
                perturbed_loss = loss_fn(perturbed_output, target)

                if overfit_gap is not None and overfit_gap > 0:
                    perturbed_loss -= overfit_gap * confidence_penalty(perturbed_output, beta=BETA)

                if method == "fedprox" and server_model is not None:
                    prox_term = 0.0
                    for  param, server_param in zip(model.parameters(), server_model.parameters()):
                        prox_term += torch.sum((param - server_param.detach()) ** 2)
                    perturbed_loss += (mu / 2.0) * prox_term

                elif method == "cosine" and server_model is not None:
                    perturbed_loss += mu * cosine_regularization(model, server_model)


                perturbed_loss.backward()
                optimizer.second_step(zero_grad=True)

               

            else:
                with autocast():
                    output = model(data)
                    loss = loss_fn(output, target)
    
                    if method == "fedprox" and server_model is not None:
                        prox_term = 0.0
                        prox_term += torch.sum((param - server_param.detach()) ** 2)
                        loss += (mu / 2.0) * prox_term
                    elif method == "cosine" and server_model is not None:
                        loss += mu * cosine_regularization(model, server_model)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)


    acc = correct / total
    avg_loss = total_loss / (len(loader) * local_epochs)
    return avg_loss, acc


    
def fed_avg(models, weights, fedbn=False):
    new_state_dict = {}

    for key in models[0].state_dict().keys():
        if fedbn and any(x in key.lower() for x in ['bn', 'norm']):
            new_state_dict[key] = models[0].state_dict()[key]
        else:
            new_state_dict[key] = sum(
                weights[i] * models[i].state_dict()[key] for i in range(len(models))
            )

    return new_state_dict


# In[ ]:


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()

                if group["adaptive"]:
                    perturbation = (torch.abs(p) + 1e-12) * p.grad * scale.to(p)
                else:
                    perturbation = p.grad * scale.to(p)

                p.add_(perturbation)

        if zero_grad:
            self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if group["adaptive"]:
                    grad = grad * torch.abs(p)
                grads.append(grad.norm(p=2).to(device))
        return torch.norm(torch.stack(grads), p=2)


# In[ ]:


# From FedDG-GA paper (https://github.com/MediaBrain-SJTU/FedDG-GA/blob/master/utils/weight_adjust.py)

def refine_weight_dict_by_GA(weight_dict, site_before_results_dict, site_after_results_dict, step_size=0.1, fair_metric='loss'):
    if fair_metric == 'acc':
        signal = -1.0
    elif fair_metric == 'loss':
        signal = 1.0
    else:
        raise ValueError('fair_metric must be acc or loss')
    
    value_list = []
    for site_name in site_before_results_dict.keys():
        value_list.append(site_after_results_dict[site_name][fair_metric] - site_before_results_dict[site_name][fair_metric])
    
    value_list = np.array(value_list)
    
    
    step_size = 1./3. * step_size
    norm_gap_list = value_list / np.max(np.abs(value_list))
    
    for i, site_name in enumerate(weight_dict.keys()):
        weight_dict[site_name] += signal * norm_gap_list[i] * step_size

    weight_dict = weight_clip(weight_dict)
    
    return weight_dict

def weight_clip(weight_dict):
    new_total_weight = 0.0
    for key_name in weight_dict.keys():
        weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
        new_total_weight += weight_dict[key_name]
    
    for key_name in weight_dict.keys():
        weight_dict[key_name] /= new_total_weight
    
    return weight_dict


def compute_ece(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.any():
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)

    return ece.item()
def save_softmax_hist_summary(probs_tensor, bins=25):
    counts, bin_edges = np.histogram(probs_tensor.numpy().flatten(), bins=bins, range=(0, 1), density=True)
    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist()
    }


# In[ ]:


def evaluate(model, loader, loss_fn, device):
    model.eval()
    model.to(device)
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)


    return val_loss / len(loader), correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    model.to(device)
    val_loss = 0.0
    correct, total = 0, 0

    all_probs, all_labels = [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)


            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(target.cpu())

    avg_loss = val_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy, torch.cat(all_probs), torch.cat(all_labels)


# In[ ]:


run_name = "all"
log_dir = f"shared_dir/OCFL/logs_sens/{run_name}"
os.makedirs(log_dir, exist_ok=True)


log_path = os.path.join(log_dir, "training_log.txt")

logging.basicConfig(
    filename=log_path,
    filemode='w', 
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)



# In[ ]:


import copy
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
global_model = get_model().to(device)

client_models = [get_model().to(device) for _ in range(num_clients)]

# Sync all clients to global model
for model in client_models:
    model.load_state_dict(global_model.state_dict())


client_weights = [
    len(train_loaders[i].dataset) / sum(len(loader.dataset) for loader in train_loaders)
    for i in range(num_clients)
]

ROUNDS = 20
LOCAL_EPOCHS = 5


USE_FEDBN = True
METHOD = "cosine"
USE_WP = True
USE_GA = False
USE_CONFIDENCE = True
MU = 0.005
loss_fn = nn.CrossEntropyLoss()

# Initialize per-client optimizers 
if USE_WP:
    optimizers = [
        SAM(params=client_models[i].parameters(),
                base_optimizer=optim.SGD,
                lr=0.01,
                weight_decay=0.0001
                rho=0.05,  
                momentum=0.9)
        for i in range(num_clients)]
else:
    optimizers = [
        optim.SGD(client_models[i].parameters(), lr=0.01, momentum=0.9)
        for i in range(num_clients)]
scalers = [GradScaler() for _ in range(num_clients)]


schedulers = [
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[i], mode='max', patience=3, factor=0.5)
    for i in range(num_clients)
]

best_acc = 0.0
best_state = None

val_acc_history = []
val_loss_history = []

acc_matrix = []

######################## Generalization Adjustment ########################
weight_dict         = {site: 1/num_clients for site in site_names}  
site_results_before = {site: None for site in site_names}
site_results_after  = {site: None for site in site_names}
step_size   = 0.20      
decay_per_r = step_size / ROUNDS
########################################################################
client_train_acc_prev = {site: None for site in site_names}
client_val_acc_prev   = {site: None for site in site_names}


logging.info("HYPERPARAMETERS")
logging.info(f"METHOD: {METHOD}")
logging.info(f"USE_WP: {USE_WP}")
logging.info(f"USE_GA: {USE_GA}")
logging.info(f"USE_CONFIDENCE: {USE_CONFIDENCE}")
logging.info(f"Cosine Regularization: {'Enabled' if METHOD == 'cosine' else 'Disabled'}")
logging.info(f"USE_FEDBN: {USE_FEDBN}")
logging.info(f"MU (reg strength): {MU}")
logging.info(f"Optimizer: SGD with momentum=0.9")
logging.info(f"LR: 0.01")
logging.info(f"WP rho: 0.05")
logging.info(f"BETA: {BETA}")

    
ece_per_client = {site: [] for site in site_names}
hist_per_client = {site: [] for site in site_names}

for round in range(ROUNDS):
    print(f"\n-- Round {round} --")
    logging.info(f"\n--- Round {round} ---")
    
    site_results_before.clear()
    if USE_GA:
        site_results_after.clear()


    if USE_CONFIDENCE:
    
        overfit_gaps = {}
        for site in site_names:
            if client_train_acc_prev[site] is not None and client_val_acc_prev[site] is not None:
                gap = client_train_acc_prev[site] - client_val_acc_prev[site]
                if gap > 0:
                    overfit_gaps[site] = gap

        k = max(1, int(0.5 * len(site_names)))  
        top_k_sites = sorted(overfit_gaps.items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_sites = {site for site, _ in top_k_sites}
        print(f"Top-k overfitting sites: {top_k_sites}")
        

    if USE_GA:
        site_results_before = {}
    for i in range(num_clients):
        overfit_gap = None
        penalize = False
        if USE_CONFIDENCE:
            site = site_names[i]
            penalize = site in top_k_sites
            overfit_gap = overfit_gaps.get(site, None)
            

        server_model = copy.deepcopy(global_model)
        train_loss, train_acc = local_train(
            model=client_models[i],
            loader=train_loaders[i],
            loss_fn=loss_fn,
            optimizer=optimizers[i],
            device=device,
            scaler=scalers[i],
            local_epochs=LOCAL_EPOCHS,
            method=METHOD,
            server_model=server_model,  
            mu=MU,
            use_wp=USE_WP,
            overfit_gap=overfit_gap if penalize else None
        )


        print(f"Client {site_names[i]} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logging.info(f"Client {site_names[i]} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        if USE_CONFIDENCE:
            client_train_acc_prev[site_names[i]] = train_acc

        
        if USE_GA:
            val_loss, val_acc = evaluate(client_models[i],  
                                     val_loaders[i],
                                     loss_fn,
                                     device)
            site_results_before[site_names[i]] = {"loss": val_loss,
                                          "acc":  val_acc}
            

    # Aggregate
    if USE_GA:
        agg_w = [weight_dict[site] for site in site_names]
    else:
        agg_w = client_weights
        
    global_state = fed_avg(client_models, agg_w, fedbn=USE_FEDBN)

    global_model.to(device) 


    global_model.load_state_dict(global_state)
    for model in client_models:
      
        if USE_FEDBN:
            model_state = model.state_dict()
            for key in global_state:
                if not any(k in key.lower() for k in ['bn', 'norm']):
                    model_state[key] = global_state[key]
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(global_state)
        model.to(device)


    # Validation
    total_acc, total_val_loss = 0.0, 0.0
    global_model.eval()
    with torch.no_grad():
        for i in range(num_clients):

            v_loss, v_acc, probs_tensor, labels_tensor = evaluate(client_models[i], val_loaders[i], loss_fn, device)
            #########################################################

            hist_data = save_softmax_hist_summary(probs_tensor, bins=25)
            hist_per_client[site_names[i]].append(hist_data)

            ece = compute_ece(probs_tensor, labels_tensor)

            ece_per_client[site_names[i]].append(ece)

            print(f"Client {site_names[i]} | ECE: {ece:.4f}")
            logging.info(f"Client {site_names[i]} | ECE: {ece:.4f}")

            #########################################################
            # store per-client result if GA is enabled
            if USE_GA:
                site_results_after[site_names[i]] = {"loss": v_loss,
                                                     "acc":  v_acc}
            
            if USE_CONFIDENCE:
                client_val_acc_prev[site_names[i]] = v_acc


            total_val_loss += v_loss
            total_acc      += v_acc
    
            print(f"Client {site_names[i]} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")
            logging.info(f"Client {site_names[i]} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")
            schedulers[i].step(v_acc)



    if USE_GA and all(site in site_results_after for site in site_names):
            acc_matrix.append([site_results_after[site]['acc'] for site in site_names])

    avg_val_acc = total_acc / num_clients
    avg_val_loss = total_val_loss / num_clients
    val_acc_history.append(avg_val_acc)
    val_loss_history.append(avg_val_loss)

    print(f"\nAvg Val Loss: {avg_val_loss:.4f} | Avg Val Acc: {avg_val_acc:.4f}")
    logging.info(f"\nAvg Val Loss: {avg_val_loss:.4f} | Avg Val Acc: {avg_val_acc:.4f}")
    
    if USE_GA: 
        weight_dict = refine_weight_dict_by_GA(
            weight_dict,
            site_results_before,
            site_results_after,
            step_size,
            fair_metric='acc'
        )

        step_size = max(step_size - decay_per_r, 0.0)


    

    if avg_val_acc > best_acc:
        # Save best client states
        best_client_states = [client_models[i].state_dict() for i in range(num_clients)]
        best_acc = avg_val_acc
        best_state = global_model.state_dict()
        torch.save({
            'server_model': best_state,
            'client_models': best_client_states,
            'best_acc': best_acc,
            'round': round,
        }, os.path.join(log_dir, "best_checkpoint.pth"))

        print(f"Saved new best model with Val Acc: {best_acc:.4f}")
        logging.info(f"Saved new best model with Val Acc: {best_acc:.4f}")

# import json

with open(os.path.join(log_dir, "ece_per_round.json"), "w") as f:
    json.dump(ece_per_client, f)
with open(os.path.join(log_dir, "hist_softmax_25bins.json"), "w") as f:
    json.dump(hist_per_client, f)




# In[ ]:



latest_client_states = [client_models[i].state_dict() for i in range(num_clients)]
latest_state = global_model.state_dict()
torch.save({
    'server_model': latest_state,
    'client_models': latest_client_states,
    'round': round,
}, os.path.join(log_dir, "latest_checkpoint.pth"))

import matplotlib.pyplot as plt

rounds = list(range(1, ROUNDS + 1))

plt.figure()
plt.plot(rounds, val_acc_history, marker='o')
plt.title("Validation Accuracy per Round")
plt.xlabel("Federated Round")
plt.ylabel("Avg Val Accuracy")
plt.grid(True)
plt.savefig(os.path.join(log_dir, "val_accuracy_curve.png"))
plt.close()

plt.figure()
plt.plot(rounds, val_loss_history, marker='o', color='red')
plt.title("Validation Loss per Round")
plt.xlabel("Federated Round")
plt.ylabel("Avg Val Loss")
plt.grid(True)
plt.savefig(os.path.join(log_dir, "val_loss_curve.png"))
plt.close()


# In[ ]:


def evaluate_checkpoint(checkpoint_path, label="Checkpoint"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    global_model.load_state_dict(checkpoint['server_model'])
    client_states = checkpoint['client_models']
    round_num = checkpoint.get('round', -1)

    print(f"\nEvaluation on Test Sets ({label}, Round {round_num})")
    logging.info(f"\nEvaluation on Test Sets ({label}, Round {round_num})")

    total_acc = 0.0
    for i in range(num_clients):
        client_models[i].load_state_dict(client_states[i])
        client_models[i].eval()
        correct, total = 0, 0
        test_loss = 0.0
        test_loader = test_loaders[i]

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = client_models[i](x)
                loss = loss_fn(output, y)
                test_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        total_acc += acc
        test_loss /= len(test_loader)
        print(f"Client {site_names[i]} | Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f}")
        logging.info(f"Client {site_names[i]} | Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f}")

    avg_acc = total_acc / num_clients
    print(f"Average Test Accuracy ({label}): {avg_acc:.4f}\n")
    logging.info(f"Average Test Accuracy ({label}): {avg_acc:.4f}")



evaluate_checkpoint(os.path.join(log_dir, "best_checkpoint.pth"), label="Best")
evaluate_checkpoint(os.path.join(log_dir, "latest_checkpoint.pth"), label="Latest")


# In[ ]:




