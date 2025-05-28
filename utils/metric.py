import math
import torch

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance, pearsonr, spearmanr, entropy, ks_2samp, cramervonmises_2samp

#################################################################################################################

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return entropy(p, q)

def js_divergence(p, q):
    return jensenshannon(p, q) ** 2

def distribution_metrics(y_true_np, y_pred_np, bins=50, range=None):
    hist_true, _ = np.histogram(y_true_np, bins=bins, range=range, density=False)
    hist_pred, _ = np.histogram(y_pred_np, bins=bins, range=range, density=False)

    hist_true = hist_true.astype(np.float64)
    hist_pred = hist_pred.astype(np.float64)

    hist_true += 1e-10
    hist_pred += 1e-10
    hist_true /= hist_true.sum()
    hist_pred /= hist_pred.sum()

    return {
        'wasserstein': wasserstein_distance(y_true_np, y_pred_np),
        'kl': kl_divergence(hist_true, hist_pred),
        'js': js_divergence(hist_true, hist_pred),
        'ks': ks_2samp(y_true_np, y_pred_np).statistic,
        'cvm': cramervonmises_2samp(y_true_np, y_pred_np).statistic,
        'tv': total_variation_distance(hist_true, hist_pred)
    }

def group_distribution_metrics(y_true_np, y_pred_np, sensitive_attr_np, bins=50, range=None):
    groups = np.unique(sensitive_attr_np.astype(int))
    if len(groups) != 2:
        raise ValueError(f"필요한 그룹(0, 1) 중 일부가 누락되었습니다. 존재하는 그룹: {groups}")

    results = {}
    metrics = ['wasserstein', 'kl', 'js', 'ks', 'cvm', 'tv']
    group_metrics = {}

    for g in groups:
        idx = sensitive_attr_np == g
        group_metrics[f"group_{g}"] = distribution_metrics(y_true_np[idx], y_pred_np[idx], bins=bins, range=range)

    for m in metrics:
        try:
            diff = abs(group_metrics["group_0"][m] - group_metrics["group_1"][m])
            results[f"{m}_diff"] = diff
            results[f"{m}_g0"] = group_metrics["group_0"][m]
            results[f"{m}_g1"] = group_metrics["group_1"][m]
        except KeyError as e:
            raise KeyError(f"그룹별 지표 계산 중 '{e}' 누락. 현재 그룹들: {group_metrics.keys()}")
    return results

def fair_metric(output, labels, sens, idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]>0

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y>0)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y>0)

    pred_y = (output[idx].squeeze()>0.5).type_as(labels).cpu().numpy()

    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity, equality

def compute_graph_fairness_stats(data, edge_index):
    sens = data.sensitive_attr.cpu().numpy()
    labels = data.y.cpu().numpy()
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    num_nodes = len(sens)

    print("========== Dataset Summary ==========")
    print(f"Total nodes: {num_nodes}")
    print(f"Total edges: {edge_index.shape[1]}")

    # 1. 민감 속성 분포
    g0, g1 = np.sum(sens == 0), np.sum(sens == 1)
    print("\n--- Sensitive Attribute Distribution ---")
    print(f"Group 0: {g0} ({g0/num_nodes:.2%})")
    print(f"Group 1: {g1} ({g1/num_nodes:.2%})")

    # 2. 예측값 분포
    print("\n--- Label Distribution ---")
    print(f"Overall: mean={labels.mean():.4f}, std={labels.std():.4f}")
    print(f"Group 0: mean={labels[sens==0].mean():.4f}, std={labels[sens==0].std():.4f}")
    print(f"Group 1: mean={labels[sens==1].mean():.4f}, std={labels[sens==1].std():.4f}")

    # 3. 상관관계
    pear, _ = pearsonr(sens, labels)
    spear, _ = spearmanr(sens, labels)
    print("\n--- Correlation (Sensitive vs Label) ---")
    print(f"Pearson:  {pear:.4f}")
    print(f"Spearman: {spear:.4f}")

    # 4. Homophily
    def homophily(attr):
        return np.mean(attr[src] == attr[dst])
    sens_hom = homophily(sens)
    label_hom = homophily(labels.round())  # 연속형 label인 경우 이진화
    print("\n--- Graph Homophily ---")
    print(f"Sensitive attribute homophily: {sens_hom:.4f}")
    print(f"Label homophily: {label_hom:.4f}")

    # 5. Group별 node degree
    degree = np.bincount(src, minlength=num_nodes)
    deg_g0 = degree[sens == 0]
    deg_g1 = degree[sens == 1]
    print("\n--- Node Degree (per group) ---")
    print(f"Group 0: mean={deg_g0.mean():.2f}, std={deg_g0.std():.2f}")
    print(f"Group 1: mean={deg_g1.mean():.2f}, std={deg_g1.std():.2f}")

    # 6. 이웃 구성 동질성 (각 노드 이웃 중 같은 그룹 비율 평균)
    same_group_ratios = []
    for i in range(num_nodes):
        neighbors = dst[src == i]
        if len(neighbors) > 0:
            same_ratio = np.mean(sens[neighbors] == sens[i])
            same_group_ratios.append(same_ratio)
    print("\n--- Neighborhood Composition ---")
    print(f"Average same-group neighbor ratio: {np.mean(same_group_ratios):.4f}")

def fair_metric_regression(output, labels, sens):
    y_g0 = output[sens == 0]
    y_g1 = output[sens == 1]
    
    # 그룹별 MSE 차이
    mse_g0 = mean_squared_error(labels[sens == 0].cpu().numpy(), y_g0.cpu().numpy()) if len(y_g0) > 0 else 0.0
    mse_g1 = mean_squared_error(labels[sens == 1].cpu().numpy(), y_g1.cpu().numpy()) if len(y_g1) > 0 else 0.0
    mse_diff = abs(mse_g0 - mse_g1)

    # 그룹별 MAE 차이
    mae_g0 = mean_absolute_error(labels[sens == 0].cpu().numpy(), y_g0.cpu().numpy()) if len(y_g0) > 0 else 0.0
    mae_g1 = mean_absolute_error(labels[sens == 0].cpu().numpy(), y_g0.cpu().numpy()) if len(y_g0) > 0 else 0.0
    mae_diff = abs(mae_g0 - mae_g1)

    # 그룹별 평균 차이
    mean_g0 = y_g0.mean().item() if len(y_g0) > 0 else 0.0
    mean_g1 = y_g1.mean().item() if len(y_g1) > 0 else 0.0
    mean_diff = abs(mean_g0 - mean_g1)
    
    return mse_diff, mae_diff, mean_diff

def output_fairness(preds, sens):
    p0 = preds[sens == 0].cpu().numpy()
    p1 = preds[sens == 1].cpu().numpy()

    return {
        "mean_gap": abs(p0.mean() - p1.mean()),
        "mae_gap": abs(np.mean(abs(p0 - p0.mean())) - np.mean(abs(p1 - p1.mean()))),
        "mse_gap": abs(np.mean((p0 - p0.mean()) ** 2) - np.mean((p1 - p1.mean()) ** 2)),
        "wasserstein": wasserstein_distance(p0, p1),
        "js_divergence": jensenshannon(np.histogram(p0, bins=30, density=True)[0] + 1e-10,
                                       np.histogram(p1, bins=30, density=True)[0] + 1e-10)
    }

def metric_wd(feature, adj_norm, flag, weakening_factor, max_hop):
    feature = (feature / feature.norm(dim=0)).detach().cpu().numpy()
    adj_norm = (0.5 * adj_norm + 0.5 * sparse.eye(adj_norm.shape[0])).toarray()  # lambda_{max} = 2
    emd_distances = []
    cumulation = np.zeros_like(feature)

    if max_hop == 0:
        cumulation = feature
    else:
        for i in range(max_hop):
            cumulation += pow(weakening_factor, i) * adj_norm.dot(feature)

    for i in range(feature.shape[1]):
        class_1 = cumulation[torch.eq(flag, 0), i]
        class_2 = cumulation[torch.eq(flag, 1), i]
        emd = wasserstein_distance(class_1, class_2)
        emd_distances.append(emd)

    emd_distances = [0 if math.isnan(x) else x for x in emd_distances]

    if max_hop == 0:
        print('Attribute bias : ')
    else:
        print('Structural bias : ')

    print("Sum of all Wasserstein distance value across feature dimensions: " + str(sum(emd_distances)))
    print("Average of all Wasserstein distance value across feature dimensions: " + str(np.mean(np.array(emd_distances))))

    sns.distplot(np.array(emd_distances).squeeze(), rug=True, hist=True, label='EMD value distribution')
    plt.legend()
    # plt.show()

    num_list1 = emd_distances
    x = range(len(num_list1))

    plt.bar(x, height=num_list1, width=0.4, alpha=0.8, label="Wasserstein distance on reachability")
    plt.ylabel("Wasserstein distance")
    plt.legend()
    # plt.show()

    return emd_distances