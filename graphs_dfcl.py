import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn,optim
from torch.distributions.multivariate_normal import MultivariateNormal as MN
from torch.utils.data import TensorDataset,DataLoader,Dataset
from sympy import symbols, solve, Symbol, lambdify
from scipy.optimize import fsolve, minimize_scalar
from os.path import exists as file_exists
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import time
from scipy.optimize import minimize,NonlinearConstraint
import copy
import shutil
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

# Helper Functions
def high_utility(d,args):
    ql,qh = args
    Uh = (4*((qh+ql)**2)*(qh-d))/(4*qh+3*ql-d)**2
    return Uh

def nb_utility(beta,models,data,fixed_q,default_utilities,flip):
    utilities = [compute_utility([q(beta,models,data),fixed_q],firm,flip)[0]-default_utilities[i] for i,firm in enumerate(firms)]
    print('utilities',-utilities[0]*utilities[1])
    return -utilities[0]*utilities[1]

def compute_utility(ql,qh,firm):
    if firm=='low':
        return (theta**2)*ql*qh*(qh-ql)/((4*qh-ql)**2)
    if firm=='high':
        return 4*(theta**2)*(qh**2)*(qh-ql)/((4*qh-ql)**2)

def q(loss):
    return c-loss

def nash_obj(ql, qh, init_q):
    return -((compute_utility(ql, qh, 'high') - compute_utility(*init_q, 'high')) *  
            (compute_utility(ql, qh, 'low') - compute_utility(*init_q, 'low')))

def max_ratio(opt_qh, init_q):
    opt_data = minimize_scalar(nash_obj, args=(opt_qh, init_q), bounds=(0, 1), method='bounded')
    opt_ql = opt_data.x
    return opt_ql / c

def ratio_obj(c, U_l0, U_h0):
    return -(4*(1-c)/((4-c)**2)-U_h0) * (c*(1-c)/((4-c)**2)-U_l0)

def compute_firm_utilities(firm, x_firm):
    ql_array = np.linspace(0, 1, 25)
    qh_array = np.linspace(0, 1, 25)
    if x_firm == 'low':
        utilities = [[compute_utility(ql, qh, firm) for ql in ql_array] for qh in qh_array]
    elif x_firm == 'high':
        utilities = [[compute_utility(ql, qh, firm) for qh in qh_array] for ql in ql_array]
    if firm=='low':
        graph_data = ql_array
        cbar_data = qh_array
    elif firm=='high':
        graph_data = qh_array
        cbar_data = ql_array
    plot_data = [np.column_stack([graph_data, u]) for u in utilities]
    plots = LineCollection(plot_data, array=cbar_data, cmap='plasma')
    return plots

def find_intersection(x, y):
    if np.any(np.sign(x-y) < 0):
        return np.argwhere(np.sign(x - y) == -1).flatten()[0]
    return np.nan

# Figure 8
def nash_max():
    init_vals = []
    nash_vals = []
    ql_vals = np.linspace(0, 1, 10)
    qh_vals = np.linspace(0, 1, 10)
    x = np.linspace(0, 1, 1000)
    opt_vals = []
    fig, ax = plt.subplots()
    for i, ql in enumerate(ql_vals):
        for j in range(i, len(qh_vals)):
            if np.isnan(ql/qh_vals[j]):
                continue
            U_h0 = compute_utility(ql, qh_vals[j], 'high')
            U_l0 = compute_utility(ql, qh_vals[j], 'low')
            opt_data = minimize_scalar(ratio_obj, args=(U_l0, U_h0), bounds=(0, 1), method='bounded')    
            if opt_data.x < (ql/qh_vals[j]):
                continue
            if compute_utility(opt_data.x, 1, 'high') < U_h0 or compute_utility(opt_data.x, 1, 'low') < U_l0:
                continue
            opt_vals.append([opt_data.x, -ratio_obj(opt_data.x, U_l0, U_h0)])
            y = [-ratio_obj(x_val, U_l0, U_h0) for x_val in x]
            init_vals.append((ql, qh_vals[j]))
            nash_vals.append(y)
    
    sorted_idxs = np.argsort(list(map(lambda x: x[0]/x[1], init_vals)))
    sorted_init_vals = np.array(list(map(lambda x: x[0]/x[1], init_vals)))[sorted_idxs]
    sorted_nash_vals = np.array(nash_vals)[sorted_idxs]
    plot_data = [np.column_stack([x, n]) for n in sorted_nash_vals]
    plots = LineCollection(plot_data, array=sorted_init_vals, cmap='plasma')
    ax.add_collection(plots)
    y_bounds = [-0.004, 0.004]
    ax.set_ylim(y_bounds)
    cbar = fig.colorbar(plots, label='$q_{l,0}/q_{h,0}$')
    cbar.set_label(label='$q_{l,0}/q_{h,0}$', size=label_fs)
    ax.set_xlabel('$q_{l}$', fontsize=label_fs)
    ax.set_ylabel('$N(q_{l}, q_{h}=1)$', fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=axis_fs)
    plt.scatter(np.transpose(opt_vals)[0], np.transpose(opt_vals)[1], c='g', marker='o', s=20, zorder=10)
    plt.axvline(x=0.43, ymin=0, ymax=1, ls='--', label='$q_l=0.43$')
    plt.legend(fontsize=legend_fs)
    plt.sci(plots)
    sns.despine()
    fig.tight_layout()
    plt.savefig(f'{plot_dir}nash_max.pdf')
    plt.show()

def compute_q_ratio(prev_r, qh_r):
    return 4 - (((4-prev_r)**2)/(2*(1-prev_r)))*(qh_r-np.sqrt((qh_r**2)-((12*(1-prev_r))/((4-prev_r)**2))*qh_r))

# Figure 5
def plot_ratio():
    fig, ax = plt.subplots()
    prev_r_array = np.linspace(0, 0.99, 1000)
    qh_r_upper_lim = 100
    qh_r_array = np.linspace(1, qh_r_upper_lim, 100)
    ratios = [[compute_q_ratio(prev_r, qh_r) for prev_r in prev_r_array] for qh_r in qh_r_array]
    plot_data = [np.column_stack([prev_r_array, r]) for r in ratios]
    plots = LineCollection(plot_data, array=qh_r_array, cmap='plasma')
    ax.add_collection(plots)
    cbar = fig.colorbar(plots)
    cbar.set_label(label='b', size=label_fs)
    cbar_labels = ['1'] + [(i+1) * int(qh_r_upper_lim/5) for i in range(5)]
    cbar.set_ticks(np.linspace(1, qh_r_upper_lim, len(cbar_labels)))
    cbar.ax.set_yticklabels(cbar_labels, fontsize=axis_fs)
    plt.sci(plots)
    ax.set_xlabel('a', fontsize=label_fs)
    ax.set_ylabel('B(a,b)', fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=axis_fs)
    fig.tight_layout()
    sns.despine()
    plt.savefig(f'{plot_dir}ratio.pdf')
    plt.show()

# Figure 4
def plot_firm_utilities():
    firms = ['low', 'high']
    x_firms = ['low', 'high']
    utilities = []
    fig, axs = plt.subplots(len(firms), len(x_firms), figsize=(8, 5))
    for i, firm in enumerate(firms):
        for j, x_firm in enumerate(x_firms):
            plots = compute_firm_utilities(firm=firms[i], x_firm=x_firms[j])
            axs[i][j].add_collection(plots)
            
            cbar = fig.colorbar(plots, ax=axs[i][j])
            cbar_labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            cbar.set_ticks(np.linspace(0, 1, 6))
            cbar.ax.set_yticklabels(cbar_labels, fontsize=axis_fs)
            
            x_label = '$q_l$' if x_firm == 'low' else '$q_h$'
            y_label = '$U_l(q_l, q_h)$' if firm == 'low' else '$U_h(q_l, q_h)$'
            bar_label = '$q_h$' if x_firm == 'low' else '$q_l$'
            ylim = 0.025 if firm == 'low' else 0.3
            cbar.set_label(label=bar_label, size=label_fs)
            
            axs[i][j].set_ylim([0, ylim])
            axs[i][j].set_xlabel(x_label, fontsize=label_fs)
            axs[i][j].set_ylabel(y_label, fontsize=label_fs)
            axs[i][j].tick_params(axis='both', labelsize=axis_fs)
    
    fig.autofmt_xdate()
    fig.tight_layout()
    sns.despine()
    plt.savefig(f'{plot_dir}firm_utilities.pdf')
    plt.show()


# Figure 7
def plot_convergence():
    fig, ax = plt.subplots()
    init_ql_vals = np.linspace(0, .99, 25)
    init_qh_vals = np.linspace(0, .99, 25)
    bounds = [[] for i in range(len(init_ql_vals))]
    for i, init_ql in enumerate(init_ql_vals):
        for j, init_qh in enumerate(init_qh_vals):
            if j <= i:
                bounds[i].append(np.nan)
            else:
                rho_star = max_ratio(1, [init_ql, init_qh])
                if (init_qh * (10 ** ((rho_star - (init_ql / init_qh)) / (4 - (5 * rho_star)))) > 1):
                    print(init_ql, init_qh, init_qh * (10 ** ((rho_star - (init_ql / init_qh)) / (4 - (5 * rho_star)))))
                bounds[i].append(init_qh * (10 ** ((rho_star - (init_ql / init_qh)) / (4 - (5 * rho_star)))))
    plot_data = [np.column_stack([init_qh_vals, bound]) for bound in bounds]
    plots = LineCollection(plot_data, array=init_ql_vals, cmap='plasma')
    ax.add_collection(plots)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 2])
    ax.set_xlabel('$q_{h,0}$', fontsize=label_fs)
    ax.set_ylabel('$q_{h,0} \\cdot 10^{\\frac{\\rho^*-\\rho_0}{4-5\\rho^*}}$', fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=axis_fs)
    plt.axhline(y=1, xmin=0, xmax=1, ls='--', label='$q_h^*=1$')
    cbar = fig.colorbar(plots)
    cbar.set_label(label='$q_{l,0}$', size=label_fs)
    plt.legend(fontsize=legend_fs)
    plt.sci(plots)
    fig.autofmt_xdate()
    fig.tight_layout()
    sns.despine()
    plt.savefig(f'{plot_dir}convergence.pdf')
    plt.show()

# Figure 6
def plot_ratio_bounds():
    fig, ax = plt.subplots()
    prev_r_array = np.linspace(0, 1, 15)
    qh_r_upper_lim = 5
    qh_r_array = np.linspace(1, qh_r_upper_lim, 100000)
    ratio_deltas = [[compute_q_ratio(prev_r, qh_r) - prev_r for qh_r in qh_r_array] for prev_r in prev_r_array]
    ratio_delta_lower_bounds = [(4 - 5 * prev_r) * np.log10(qh_r_array) for prev_r in prev_r_array]
    ratio_delta_data = [np.column_stack([qh_r_array, r]) for r in ratio_deltas]
    ratio_delta_plots = LineCollection(ratio_delta_data, array=prev_r_array, cmap='plasma')
    ratio_delta_lower_bounds_data = [np.column_stack([qh_r_array, r]) for r in ratio_delta_lower_bounds]
    ratio_delta_lower_bounds_plots = LineCollection(ratio_delta_lower_bounds_data, array=prev_r_array, cmap='plasma', linestyle='dashed')
    intersection_data = []
    for i in range(len(prev_r_array)):
        ratio_delta_data[i][0][1] = 0
        idx = find_intersection(np.transpose(ratio_delta_data[i])[1], np.transpose(ratio_delta_lower_bounds_data[i])[1])
        if np.isnan(idx):
            intersection_data.append(np.array([np.nan, np.nan]))
        else:
            intersection_data.append(ratio_delta_data[i][idx])
    b_array = np.transpose(intersection_data)[0]
    b_tilde = np.nanmin(b_array)
    rho_b_tilde = prev_r_array[np.nanargmin(b_array)]
    plt.scatter(np.transpose(intersection_data)[0],
                np.transpose(intersection_data)[1], c='g', marker='o', s=20,
                zorder=10, label='$b_a$')
    ax.vlines(x=b_tilde, ymin=0,
              ymax=np.transpose(intersection_data)[1][np.nanargmin(b_array)],
              ls='--', label='$\\tilde{b}_a\\approx$'+f'{np.round(b_tilde,
              decimals=2)}'+'$(a\\approx$'+f'{np.round(rho_b_tilde,
                                                       decimals=2)})',
              zorder=20, color='black')
    ax.add_collection(ratio_delta_plots)
    ax.set_xlim([1, 1.2])
    ax.set_ylim([0, 0.4])
    cbar = fig.colorbar(ratio_delta_plots)
    cbar.set_label(label='a', size=label_fs)
    plt.sci(ratio_delta_plots)
    ax.set_xlabel('b', fontsize=label_fs)
    ax.set_ylabel('B(a,b) - a', fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=axis_fs)
    plt.legend(fontsize=legend_fs)

    fig.autofmt_xdate()
    fig.tight_layout()
    sns.despine()
    plt.savefig(f'{plot_dir}ratio_bounds.pdf')
    plt.show()

if __name__=='__main__':
    np.random.seed(1)
    torch.manual_seed(1)

    c = 1
    theta = 1
    label_fs = 16
    axis_fs = 14
    legend_fs = 12
    plot_dir = 'datasharing plots/'

    # Generate Plots
    #plot_convergence()    
    #plot_firm_utilities()
    #plot_ratio()
    #nash_max()
    #plot_ratio_bounds()
