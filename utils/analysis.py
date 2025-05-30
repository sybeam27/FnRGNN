import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt

from torch_geometric.utils import degree
from scipy.stats import wasserstein_distance
from scipy.stats import ttest_ind, levene, ks_2samp
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix

##########################################################

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.power(rowsum, -1.0, where=rowsum != 0)
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def homophily_ratio(edge_index, sensitive_attr):
    edge_index = edge_index.cpu().numpy()
    sens = sensitive_attr.cpu().numpy()
    same = sum(sens[u] == sens[v] for u, v in zip(*edge_index))
    return same / edge_index.shape[1]

def assortativity_coefficient(data):
    G = to_networkx(data, to_undirected=True)
    nx.set_node_attributes(G, {i: int(data.sensitive_attr[i]) for i in range(data.num_nodes)}, "sensitive")
    return nx.attribute_assortativity_coefficient(G, "sensitive")

def local_neighborhood_fairness(edge_index, sensitive_attr):
    edge_index = edge_index.cpu().numpy()
    sens = sensitive_attr.cpu().numpy()
    diffs = []

    for node in range(len(sens)):
        neighbors = edge_index[1][edge_index[0] == node]
        if len(neighbors) > 0:
            local = sens[neighbors].mean()
            global_ = sens.mean()
            diffs.append(abs(local - global_))
    return np.mean(diffs)

def degree_balance(edge_index, sensitive_attr):
    num_nodes = sensitive_attr.size(0)
    degrees = np.bincount(edge_index[0].cpu().numpy(), minlength=num_nodes)
    sens = sensitive_attr.cpu().numpy()
    group0_degrees = degrees[sens == 0]
    group1_degrees = degrees[sens == 1]
    return abs(group0_degrees.mean() - group1_degrees.mean())

def structural_bias(features, edge_index, sens, num_hops=2, alpha=0.9):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=features.size(0)).tocsr()
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    feat = features.cpu().numpy()
    feat_smooth = feat.copy()
    
    for _ in range(num_hops):
        feat_smooth = alpha * adj.dot(feat_smooth) + (1 - alpha) * feat
    
    sens = sens.cpu().numpy()
    s0 = feat_smooth[sens == 0]
    s1 = feat_smooth[sens == 1]
    emd = np.mean([wasserstein_distance(s0[:, i], s1[:, i]) for i in range(s0.shape[1])])

    return emd

def analyze_structural_bias_with_tests(data, df, sens_attr='sens', y_col='y', y_pred_col='y_pred'):
    df['degree'] = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
    df['error'] = np.abs(df[y_col] - df[y_pred_col])
    summary = df.groupby(sens_attr).agg({
        'degree': ['mean', 'std'],
        'error': ['mean', 'std'],
        y_pred_col: ['var']
    }).round(3)

    print(summary)

    g0 = df[df[sens_attr] == 0]
    g1 = df[df[sens_attr] == 1]
    t_stat, p_ttest = ttest_ind(g0['error'], g1['error'], equal_var=False)
    w_stat, p_levene = levene(g0['error'], g1['error'])
    ks_stat, p_ks = ks_2samp(g0[y_pred_col], g1[y_pred_col])

    print(f"  • t-test:        p = {p_ttest:.4f}")
    print(f"  • Levene test:   p = {p_levene:.4f}")
    print(f"  • KS test:       p = {p_ks:.4f}")

    return {
        'summary': summary,
        'p_ttest_error_mean': p_ttest,
        'p_levene_error_var': p_levene,
        'p_ks_pred_dist': p_ks
    }

def plot_all_runs(logs, ds_name):
    plt.figure(figsize=(5, 3))
    for run_id, run_log in logs[ds_name].items():
        val_scores = run_log['val_score']
        plt.plot(range(len(val_scores)), val_scores, label=f'Run {run_id}')
    
    plt.title(f"Validation Score (MSE + Wasserstein) for Dataset: {ds_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_total_loss_all_runs(logs, ds_name):
    plt.figure(figsize=(5, 3))
    for run_id, run_log in logs[ds_name].items():
        total_losses = run_log['train_loss']['total']
        plt.plot(range(len(total_losses)), total_losses, label=f'Run {run_id}')
    
    plt.title(f"Training Total Loss for Dataset: {ds_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_metrics(logs, ds_name):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    for run_id, run_log in logs[ds_name].items():
        val_scores = run_log['val_score']
        total_losses = run_log['train_loss']['total']
        axs[0].plot(range(len(total_losses)), total_losses, label=f'Run {run_id}')
        axs[1].plot(range(len(val_scores)), val_scores, label=f'Run {run_id}')
    
    axs[0].set_title(f"Total Training Loss - {ds_name}")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Total Loss")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title(f"Validation Score (MSE + Wasserstein) - {ds_name}")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Val Score")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

