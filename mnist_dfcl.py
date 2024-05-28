import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal as MN
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset, Subset
from sympy import symbols,solve
from scipy.optimize import fsolve, minimize_scalar
from os.path import exists as file_exists
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import time
from scipy.optimize import minimize, NonlinearConstraint
import copy
import shutil

'''
Data Functions
'''
class custom_dataset(Dataset):
    def __init__(self, dataset, indices, labels):
        self.dataset = Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)
    def __len__(self):
        return len(self.targets)

def get_train_data(firm, cur_datasets):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=torch.tensor([0.1307]),
                                                         std=torch.tensor([0.3081]))])
    train_dataset = datasets.MNIST(root="./datasets/", train=True, download=True, transform=transform)
    if cur_datasets:
        low_dataset = cur_datasets[0].dataset
        high_dataset = cur_datasets[1].dataset
        count_threshold = int(min(torch.add(torch.unique(low_dataset.targets,
                                                         return_counts=True)[1],
                                            torch.unique(high_dataset.targets,
                                                         return_counts=True)[1])))
        idxs = np.arange(num_classes * count_threshold)
        np.random.shuffle(idxs)

    else:
        idx_subsets = [(torch.tensor(train_dataset.targets) <= label_threshold).nonzero().squeeze(),
                       (torch.tensor(train_dataset.targets) > label_threshold).nonzero().squeeze()]

        idx_props = data_props[firm]

        idxs = torch.empty(0)
        for i in range(len(idx_subsets)):
            idxs = torch.cat((idxs, idx_subsets[i][:int(idx_props[i] * len(train_dataset))])).int()

        idxs = idxs.detach().numpy()
        np.random.shuffle(idxs)
        idxs = idxs[:num_train_pts[firm]]

    train_dataset_subset = custom_dataset(train_dataset, idxs, torch.tensor(train_dataset.targets)[idxs])
    train_loader = DataLoader(dataset=train_dataset_subset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_test_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=torch.tensor([0.1307]),
                                                         std=torch.tensor([0.3081]))])
    test_dataset = datasets.MNIST(root="./datasets", train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader

'''
Model Functions
'''
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
                nn.Tanh(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                nn.Tanh(),
                nn.MaxPool2d(2, 2)
            )
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=400, out_features=120),
                nn.Tanh(),
                nn.Linear(in_features=120, out_features=84),
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=num_classes),
            )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def evaluate_model(path,model,data):
    if path:
        model = load_model(path)
    model.eval()
    loss = []
    acc = []
    for i,(x,y) in enumerate(data):
        with torch.no_grad():
            preds = torch.argmax(model(x.float()),dim=1)
            loss.append(~preds.eq(y))
            acc.append(preds.eq(y))
    loss = torch.cat(loss,dim=0)
    acc = torch.cat(acc,dim=0)
    return loss.float().mean().unsqueeze(0)

def load_model(path):
    model = LeNet()
    model.load_state_dict(torch.load(path))
    return model

def initialize_models(data, alphas):
    for i, firm in enumerate(firms):
        print(f'initializing {firm} model')
        model=LeNet()
        torch.save(model.state_dict(), init_paths[i])
        alpha = alphas[i]
        for j in range(num_init_train_rnds[firm]):
            update_model(path=init_paths[i], model=None, data=data[i], alpha=alpha)
    return 

def update_model(path, model, data, alpha):
    if not model:
        model = load_model(path)
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)
    loss_obj = nn.CrossEntropyLoss()
    model.train()
    for (x, y) in data:
        optimizer.zero_grad()
        loss = loss_obj(model(x.float()), y.long())
        loss.backward()
        optimizer.step()
    if path:
        torch.save(model.state_dict(), path)
    return model

def share_update_model(path, model, data, alpha):
    if not model:
        model = load_model(path)
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)
    loss_obj_l = nn.CrossEntropyLoss()
    loss_obj_h = nn.CrossEntropyLoss()
    model.train()
    for (xl, yl), (xh, yh) in zip(*data):
        optimizer.zero_grad()
        loss_l = loss_obj_l(model(xl.float()), yl.long())
        loss_h = loss_obj_h(model(xh.float()), yh.long())
        loss = (loss_l + loss_h) / 2
        loss.backward()
        optimizer.step()
    if path:
        torch.save(model.state_dict(), path)
    return model

def q(loss):
    return 1 - loss

'''
Utility Quantities
'''
def compute_utility(ql, qh, firm):
    if firm=='low':
        return ql * qh * (qh - ql) / ((4 * qh - ql) ** 2)
    if firm=='high':
        return 4 * (qh ** 2) * (qh - ql) / ((4 * qh - ql) ** 2)

def neg_nash_obj(ql, qh, init_q):
    return -((compute_utility(ql, qh, 'low') - compute_utility(*init_q, 'low')) *  
             (compute_utility(ql, qh, 'high') - compute_utility(*init_q, 'high')))

def neg_ratio_obj(r, U_l0, U_h0):
    return - ((r * (1 - r) / ((4 - r) ** 2)) - U_l0) * ((4 * (1 - r) / ((4 - r) ** 2)) - U_h0)

def compute_nash_optimum(init_q):
    U_l0 = compute_utility(*init_q, 'low')
    U_h0 = compute_utility(*init_q, 'high')
    opt_data = minimize_scalar(neg_ratio_obj, args=(U_l0, U_h0), bounds=(0, 1), method='bounded')    
    return opt_data.x, -opt_data.fun

'''
Sharing Algorithms
'''
def complete_sharing_mechanism():
    train_data = [get_train_data(firm, cur_datasets=None) for firm in firms]
    test_data = get_test_data()
    initialize_models(train_data, alphas=[init_lr, init_lr])
    qualities = [[q(evaluate_model(path=init_paths[i], model=None, data=test_data)) for i in range(len(firms))]]
    utilities = [[compute_utility(*qualities[-1], firm) for firm in firms]]
    for i, firm in enumerate(firms):
        shutil.copyfile(init_paths[i], paths[i])
    h_model = load_model(paths[1])
    l_model = load_model(paths[0])
    for t in range(T):
        print('t', t)
        
        # h model update
        share_update_model(path=paths[1], model=None, data=train_data, alpha=complete_sharing_lr)
        upd_qh = q(evaluate_model(path=paths[1], model=None, data=test_data))

        # l model update
        share_update_model(path=paths[0], model=None, data=train_data, alpha=complete_sharing_lr)
        upd_ql = q(evaluate_model(path=paths[0], model=None, data=test_data))
        
        qualities.append([upd_ql, upd_qh])
        utilities.append([compute_utility(*qualities[-1],firm) for firm in firms])
    
    torch.save(qualities, f'{plot_vals_dir}qualities_{mechanism_type}')
    torch.save(utilities, f'{plot_vals_dir}utilities_{mechanism_type}')
    return qualities, utilities

def one_sided_sharing_mechanism(share_firm):
    train_data = [get_train_data(firm, cur_datasets=None) for firm in firms]
    test_data = get_test_data()
    initialize_models(train_data, alphas=[init_lr, init_lr])
    qualities = [[q(evaluate_model(path=init_paths[i], model=None, data=test_data)) for i in range(len(firms))]]
    utilities = [[compute_utility(*qualities[-1], firm) for firm in firms]]
    for i, firm in enumerate(firms):
        shutil.copyfile(init_paths[i], paths[i])
    for t in range(T):
        print('t', t)
        
        if share_firm == 'high':
            share_path_idx = 1
            alone_path_idx = 0
        elif share_firm == 'low':
            share_path_idx = 0
            alone_path_idx = 1

        # share model update
        share_update_model(path=paths[share_path_idx], model=None, data=train_data, alpha=one_sided_sharing_lr)
        upd_share_q = q(evaluate_model(path=paths[share_path_idx], model=None, data=test_data))
        
        # alone model update
        update_model(path=paths[alone_path_idx], model=None, data=train_data[alone_path_idx], alpha=one_sided_sharing_lr)
        upd_alone_q = q(evaluate_model(path=paths[alone_path_idx], model=None, data=test_data))
        
        if share_firm == 'high':
            upd_qh = upd_share_q
            upd_ql = upd_alone_q
        elif share_firm == 'low':
            upd_ql = upd_share_q
            upd_qh = upd_alone_q

        qualities.append([upd_ql, upd_qh])
        utilities.append([compute_utility(*qualities[-1], firm) for firm in firms])
    torch.save(qualities, f'{plot_vals_dir}qualities_{mechanism_type}_{share_firm}')
    torch.save(utilities, f'{plot_vals_dir}utilities_{mechanism_type}_{share_firm}')
    return qualities, utilities

def nash_sharing_mechanism():
    train_data = [get_train_data(firm, cur_datasets=None) for firm in firms]
    test_data = get_test_data()
    initialize_models(train_data, alphas=[init_lr, init_lr])
    qualities = [[q(evaluate_model(path=init_paths[i], model=None, data=test_data)) for i in range(len(firms))]]
    utilities = [[compute_utility(*qualities[-1], firm) for firm in firms]]
    alone_qualities = [[q(evaluate_model(path=init_paths[i], model=None, data=test_data)) for i in range(len(firms))]]
    alone_utilities = [[compute_utility(*alone_qualities[-1], firm) for firm in firms]]
    ratio = qualities[-1][0] / qualities[-1][1]
    ratios = [ratio]
    optimal_ratio, optimal_nash = compute_nash_optimum(qualities[0])
    for i, firm in enumerate(firms):
        shutil.copyfile(init_paths[i], paths[i])
    h_model = load_model(paths[1])
    l_model = load_model(paths[0])
    for t in range(T):
        print('t', t)
        
        cur_ql, cur_qh = qualities[-1]

        # h model update
        share_update_model(path=paths[1], model=None, data=train_data, alpha=complete_sharing_lr)
        upd_qh = q(evaluate_model(path=paths[1], model=None, data=test_data))
        qh_ratio = upd_qh / cur_qh

        # l model update
        if ratio <= optimal_ratio and cur_ql <= optimal_ratio * max_qh:
            target_ratio = (4 - (((4-ratio)**2)/(2*(1-ratio))) * 
                             (qh_ratio - np.sqrt(qh_ratio ** 2 - ((12 * (1-ratio))/((4-ratio) ** 2)) * qh_ratio)))
            target_ql = target_ratio * upd_qh
            upd_ql = target_ql
        else:
            upd_ql = qualities[-1][0]
        
        qualities.append([upd_ql, upd_qh])
        utilities.append([compute_utility(*qualities[-1],firm) for firm in firms])
        
        ratio = upd_ql / upd_qh
        ratios.append(ratio)
    return qualities, utilities, ratios, optimal_ratio, optimal_nash

'''
Plot Functions
'''
def plot(mechanism):
    if mechanism == 'complete sharing':
        qualities, utilities = complete_sharing_mechanism()
    elif mechanism == 'one-sided sharing':
        qualities, utilities = one_sided_sharing_mechanism(share_firm)
    elif mechanism == 'nash sharing':
        qualities, utilities, ratios, optimal_ratio, optimal_nash = nash_sharing_mechanism()

    ql = torch.tensor(list(zip(*qualities))[0]).squeeze()
    qh = torch.tensor(list(zip(*qualities))[1]).squeeze()
    ul = torch.tensor(list(zip(*utilities))[0]).squeeze()
    uh = torch.tensor(list(zip(*utilities))[1]).squeeze()
    
    if mechanism == 'nash sharing':
        r = torch.tensor(ratios).squeeze()
        nash_vals = torch.tensor([-neg_nash_obj(ql_val, qh_val, qualities[0]) for (ql_val, qh_val) in zip(ql, qh)]).squeeze()
    
    x = np.linspace(1, len(ql), len(ql))

    if mechanism == 'nash sharing':
        num_rows, num_cols = (1, 4)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    else:
        num_rows, num_cols = (1, 2)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))

    palette=sns.color_palette('muted')
    
    # qualities
    sns.lineplot(x=x, y=ql, label='low-quality firm', ax=axs[0], linewidth=2, color=palette[0])
    sns.lineplot(x=x, y=qh, label='high-quality firm', ax=axs[0], linewidth=2, color=palette[1])
    if mechanism == 'nash sharing':
        sns.lineplot(x=x, y=r, label='$\\rho_t$', ax=axs[0], color=palette[2])
        sns.lineplot(x=x, y=optimal_ratio * np.ones(len(x)), label='$\\rho^*$', ls='--', ax=axs[0], color=palette[3])

    # utilities
    sns.lineplot(x=x, y=ul, label='low-quality firm', ax=axs[1], linewidth=2,
                 color=palette[0])
    sns.lineplot(x=x, y=uh, label='high-quality firm', ax=axs[1], linewidth=2,
                 color=palette[1])
    if not mechanism == 'complete sharing':
        axs[1].hlines(y=ul[-1], xmin=0, xmax=len(x), linestyles='dashed',
                      label='revenue (low-quality firm): ' + str(np.round(ul[-1].numpy(), decimals=4)), color=palette[2])
        axs[1].hlines(y=uh[-1], xmin=0, xmax=len(x), linestyles='dashed',
                      label='revenue (high-quality firm): ' + str(np.round(uh[-1].numpy(), decimals=4)), color=palette[3])
    axs[1].set_ybound(lower=-0.3, upper=0.3)

    # utility improvement and nash utility
    if mechanism == 'nash sharing':
        sns.lineplot(x=x, y=ul-ul[0], ax=axs[2], linewidth=2, label='low-quality firm')
        sns.lineplot(x=x, y=uh-uh[0], ax=axs[2], linewidth=2, label='high-quality firm')

        sns.lineplot(x=x, y=nash_vals, ax=axs[3], linewidth=2, color=palette[0])
        axs[3].hlines(y=optimal_nash, xmin=0, xmax=len(x), linestyles='dashed', linewidth=2, color=palette[1], label='$N(q_l^*,q_h^*)$')

    if mechanism == 'nash sharing':
        y_labels = ['Quality', 'Revenue', '$U - U_0$', '$N(q_l, q_h)$']
    else:
        y_labels = ['Quality', 'Revenue']

    for i in range(num_cols):
        axs[i].set_xlabel('t', fontsize=label_fs)
        axs[i].set_ylabel(y_labels[i], fontsize=label_fs)
        axs[i].tick_params(axis='both', labelsize=axis_fs)
        axs[i].legend(fontsize=legend_fs)

    sns.despine()
    fig.tight_layout()
    if mechanism == 'one-sided sharing':
        save_path = f'{plot_dir}mnist_{mechanism}_{share_firm}.pdf'
    else:
        save_path = f'{plot_dir}mnist_{mechanism}.pdf'
    plt.savefig(save_path)

def generate_plots():
    for mechanism in mechanisms:
        print(mechanism)
        plot(mechanism)
    return

if __name__=='__main__':

    # fix randomness
    np.random.seed(1)
    torch.manual_seed(1)

    # learning rates
    init_lr = 0.01
    complete_sharing_lr = 0.01
    one_sided_sharing_lr = 0.01
    hlr = 0.01

    # data parameters
    num_train_pts = {'low': 1000, 'high': 1000}

    # experiment parameters
    firms = ['low', 'high']
    label_threshold = 5
    max_qh = 1
    num_classes = 10
    batch_size = 256
    num_init_train_rnds = {'low': 0, 'high': 10} 
    data_props = {'low': [0.8, 0.2], 'high': [0.2, 0.8]}
    T = 200
    
    # directories and paths
    data_dir = 'datasets/'
    model_dir = 'models/mnist/'
    plot_dir = 'datasharing plots/'
    plot_vals_dir = 'plot_vals/mnist/'

    # plot parameters
    label_fs = 14
    axis_fs = 12
    legend_fs = 12

    # run experiments
    mechanism_type = 'complete sharing'
    mechanisms = [mechanism_type]
    
    share_firm = 'low'
    if mechanism_type == 'one-sided sharing':
        paths = [f'{model_dir}{firm}_{mechanism_type}_{share_firm}.pkl' for firm in firms]
        init_paths = [f'{model_dir}{firm}_init_{mechanism_type}_{share_firm}.pkl' for firm in firms]
    else:
        paths = [f'{model_dir}{firm}_{mechanism_type}.pkl' for firm in firms]
        init_paths = [f'{model_dir}{firm}_init_{mechanism_type}.pkl' for firm in firms]

    generate_plots()
