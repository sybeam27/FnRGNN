import math
import torch
import argparse

import numpy as np
import seaborn as sns
import scipy.sparse as sp
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from scipy import sparse
from typing import Optional
from geomloss import SamplesLoss
from torch.nn.parameter import Parameter
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor, matmul
from deeprobust.graph.defense.pgd import PGD, prox_operators
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

##############################################################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def get_sen(sens, idx_sens_train):
    num_classes = 2  # binary sensitive attribute assumed
    one_hot = F.one_hot(sens.long(), num_classes=num_classes).float()  # (N, 2)

    # training 노드에 대해서만 정규화
    group_sums = one_hot[idx_sens_train].sum(dim=0, keepdim=True)  # (1, 2)
    group_sums[group_sums == 0] = 1  # 0으로 나누는 것 방지

    one_hot[idx_sens_train] = one_hot[idx_sens_train] / group_sums  # group-normalized

    return one_hot  # shape: (N, 2)

def quantile_loss(y_true, y_pred, tau=0.9):
    error = y_true - y_pred
    return torch.mean(torch.max(tau * error, (tau - 1) * error))

def normalize_scipy(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def binarize(A_debiased, adj_ori, threshold_proportion):
    the_con1 = (A_debiased - adj_ori).A
    the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
    the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
    the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
    A_debiased = adj_ori + sp.coo_matrix(the_con1)
    assert A_debiased.max() == 1
    assert A_debiased.min() == 0
    A_debiased = normalize_scipy(A_debiased)
    return A_debiased

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# GAT 수정 필요함
class GAT_body(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_body, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
    
    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)

        return logits

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()

        self.body = GAT_body(num_layers, in_dim, num_hidden, heads, feat_drop, attn_drop, negative_slope, residual)
        self.fc = nn.Linear(num_hidden,num_classes)
    
    def forward(self, g, inputs):
        logits = self.body(g,inputs)
        logits = self.fc(logits)

        return logits

class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = self.dropout(x)
        x = self.gc2(x, edge_index)
        return x  
    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid,nclass)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        norm = 1.0 / deg[row].sqrt() / deg[col].sqrt()
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * norm.unsqueeze(1))
        return self.linear(out)

class GCN_edit(nn.Module):
    # 회귀로 수정
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        # x = self.fc(x)
        # return x

        # 회귀로 수정
        out = self.regressor(x)  # activation 없음 (linear output)
        return out.squeeze()     # [N, 1] → [N]

class GCNBody_edit(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCNBody_edit, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
def get_model(model, nfeat, num_hidden=64, dropout=0.5, num_heads=1, num_layers=1, num_out_heads=1, attn_drop=0.0, negative_slope=0.2, residual=False):
    if model == "GCN":
        model = GCN_Body(nfeat, num_hidden, dropout)
    elif model == "GAT":
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT_body(num_layers, nfeat, num_hidden, heads, dropout, attn_drop, negative_slope, residual)
    else:
        raise ValueError("Model not implemented")
    return model

# FairGNN   
class FairGNN(nn.Module):
    def __init__(self, nfeat,
                 hidden_dim=64, model='GCN', dropout=0.5, hidden=128, lr=0.001, weight_decay=1e-5, alpha=4, beta=0.01):
        super(FairGNN,self).__init__()

        nfeat = nfeat
        nhid = hidden_dim
        dropout = dropout
        
        # 추가
        self.alpha = alpha
        self.beta = beta
        
        self.estimator = GCN(nfeat, hidden, 1, dropout) # 민감 속성 추정기
        self.GNN = get_model(model, nfeat) 
        self.classifier = nn.Linear(nhid, 1) # 회귀 출력
        self.adv = nn.Linear(nhid, 1)  # 적대적 민감 속성 예측기

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=lr, weight_decay=weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr=lr, weight_decay=weight_decay)

        # self.args = args
        # self.criterion = nn.BCEWithLogitsLoss()
        # 회귀 손실 함수
        self.criterion = nn.MSELoss() 
        self.G_loss = 0
        self.A_loss = 0

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        s = self.estimator(x, edge_index) # 민감 속성 추정
        z = self.GNN(x, edge_index) # 노드 임베딩
        y = self.classifier(z) # 회귀 예측
        return y, s
    
    def optimize(self, data):
        x, edge_index = data.x, data.edge_index
        labels = data.y
        idx_train = data.idx_train
        sens = data.sensitive_attr
        idx_sens_train = data.idx_sens_train
           
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        s = self.estimator(x, edge_index)
        h = self.GNN(x, edge_index)
        y = self.classifier(h)
        s_g = self.adv(h)

        # s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score = s.detach() # 회귀용
        s_score[idx_sens_train]=sens[idx_sens_train].unsqueeze(1).float()
        # y_score = torch.sigmoid(y)
        y_score = y # 연속값 사용
        
        self.cov =  torch.abs(torch.mean((s_score - torch.mean(s_score)) * 
                                         (y_score - torch.mean(y_score))))

        self.cls_loss = self.criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
        self.adv_loss = self.criterion(s_g, s_score)                
        
        self.G_loss = self.cls_loss  + self.alpha * self.cov - self.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        
        self.A_loss = self.criterion(s_g, s_score)
        self.A_loss.backward()
        self.optimizer_A.step()

# FMP
class FMPProp(torch.nn.Module):
    _cached_sen = Optional[SparseTensor]
    def __init__(self, in_feats, out_feats, K, lambda1, lambda2, dropout=0.0, cached=False, L2=True):
        super(FMPProp, self).__init__()
        self.K = K
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L2 = L2
        self.dropout = dropout
        self.cached = cached
        self._cached_sen = None  ## sensitive matrix

        # self.propa = GraphConv(in_feats, in_feats, weight=False, bias=False, activation=None)
        self.propa = GCNConv(out_feats, out_feats, add_self_loops=False, normalize=True)  # 그래프 컨볼루션 레이어: self-loop 미포함, 메세지 전파시 degree로 normalization

    def reset_parameters(self):
        self._cached_sen = None

    def forward(self, x: Tensor, 
                edge_index, 
                idx_sens_train,
                edge_weight: OptTensor = None, 
                sens=None) -> Tensor:

        if self.K <= 0: 
            return x

        cache = self._cached_sen
        if cache is None:
            sen_mat = get_sen(sens, idx_sens_train)               ## compute sensitive matrix
            if self.cached:
                self._cached_sen = sen_mat
                self.init_z = torch.zeros((sen_mat.size()[0], x.size()[-1])).cuda()
        else:
            sen_mat = self._cached_sen # N,

        hh = x
        x = self.emp_forward(x, edge_index, hh, K=self.K, sen=sen_mat)
        
        return x

    def emp_forward(self, x, edge_index, hh, K, sen):
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma = 1/(1+lambda2)
        beta = 1/(2*gamma)

        for _ in range(K):
            if lambda2 > 0:
                y = gamma * hh + (1-gamma) * self.propa(x, edge_index)
            else:
                y = gamma * hh + (1-gamma) * x

            if lambda1 > 0:
                # z = sen @ F.softmax(y, dim=1) / (gamma * sen @ sen.t())
                y_soft = F.softmax(y, dim=1)  # (N, F)
                # Group-wise 평균 표현 벡터
                z = sen.T @ y_soft / gamma  # (C, F)
                
                # x_bar0 = sen.t() @ z
                x_bar0 = sen @ z
                x_bar1 = F.softmax(x_bar0, dim=1) ## node * feature

                correct = x_bar0 * x_bar1 
                coeff = torch.sum(correct, dim=1, keepdim=True)
                correct = correct - coeff * x_bar1

                x_bar = y - gamma * correct
                # z_bar  = z + beta * (sen @ F.softmax(x_bar, dim=1))
                z_bar = z + beta * (sen.T @ F.softmax(x_bar, dim=1)) 
                
                if self.L2:
                    z  = self.L2_projection(z_bar, lambda_=lambda1, beta=beta)
                else:
                    z  = self.L1_projection(z_bar, lambda_=lambda1)
                
                # x_bar0 = sen.t() @ z
                x_bar0 = sen @ z 
                x_bar1 = F.softmax(x_bar0, dim=1) ## node * feature
                
                correct = x_bar0 * x_bar1 
                coeff = torch.sum(correct, 1, keepdim=True)
                correct = correct - coeff * x_bar1

                x = y - gamma * correct
            else:
                x = y # z=0

            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def L1_projection(self, x: Tensor, lambda_):
        # component-wise projection onto the l∞ ball of radius λ1.
        return torch.clamp(x, min=-lambda_, max=lambda_)
    
    def L2_projection(self, x: Tensor, lambda_, beta):
        # projection on the l2 ball of radius λ1.
        coeff = (2*lambda_) / (2*lambda_ + beta)
        return coeff * x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, lambda1={}, lambda2={}, L2={})'.format(
            self.__class__.__name__, self.K, self.lambda1, self.lambda2, self.L2)
        
class FMPGNN(torch.nn.Module):
    def __init__(self, input_size, size, num_classes, num_layer, prop):
        super(FMPGNN, self).__init__()
        
        self.hidden = nn.ModuleList()
        for _ in range(num_layer-2):
            self.hidden.append(nn.Linear(size, size))

        self.first = nn.Linear(input_size, size)
        self.last = nn.Linear(size, num_classes)
        self.prop = prop

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        sens = data.sensitive_attr
        idx_sens_train = data.idx_sens_train
        
        out = F.relu(self.first(x))
        for layer in self.hidden:
            out = F.relu(layer(out))
        
        x = self.last(out)
        if sens is None:
            raise ValueError("data.sensitive_attr is None")
        x = self.prop(x, edge_index, idx_sens_train, sens=sens)  # 인자 순서 수정
        
        return x

def FMP(data, num_layers=5, lambda1=3, lambda2=3, L2=True, num_hidden=64, num_gnn_layer=2, num_classes=1, dropout=0.5, cached=False): 

    Model = FMPGNN

    prop =  FMPProp(in_feats=data.num_features,
                # out_feats=data.num_features,
                out_feats=1,
                K=num_layers, 
                lambda1=lambda1,
                lambda2=lambda2,
                dropout=dropout,
                L2=L2,
                cached=cached)

    model = Model(input_size=data.num_features, 
                  size=num_hidden, 
                #   num_classes=data.num_classes, 
                  num_classes=num_classes,  # 회귀용
                  num_layer=num_gnn_layer, 
                  prop=prop)

    return model

# GMMD
def compute_mmd(x0, x1, alpha=1.0, sample_size=500):
    def rbf_kernel(x, y, alpha):
        dist = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
        return torch.exp(-alpha * dist)

    # 샘플링 (메모리 절약)
    if x0.size(0) > sample_size:
        idx = torch.randperm(x0.size(0), device=x0.device)[:sample_size]
        x0 = x0[idx]
    if x1.size(0) > sample_size:
        idx = torch.randperm(x1.size(0), device=x1.device)[:sample_size]
        x1 = x1[idx]

    Kxx = rbf_kernel(x0, x0, alpha)
    Kyy = rbf_kernel(x1, x1, alpha)
    Kxy = rbf_kernel(x0, x1, alpha)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

class GMMDLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GMMDLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, sens):
        x = self.linear(x)
        group_0 = x[sens == 0]
        group_1 = x[sens == 1]
        if group_0.size(0) > 1 and group_1.size(0) > 1:
            mmd = compute_mmd(group_0, group_1)
        else:
            mmd = torch.tensor(0.0, device=x.device)
        return x, mmd

class GMMD(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GMMD, self).__init__()
        self.gmmd1 = GMMDLayer(in_channels, hidden_channels)
        self.gmmd2 = GMMDLayer(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, sens = data.x, data.edge_index, data.sensitive_attr
        x, mmd1 = self.gmmd1(x, edge_index, sens)
        x = F.relu(x)
        x, mmd2 = self.gmmd2(x, edge_index, sens)
        return x.squeeze(), mmd1 + mmd2

# EDITS
class X_debaising(nn.Module):
    def __init__(self, in_features):
        super(X_debaising, self).__init__()
        self.in_features = in_features
        self.s = Parameter(torch.FloatTensor(in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.s.data.uniform_(1, 1)

    def forward(self, feature):
        return torch.mm(feature, torch.diag(self.s))

class EstimateAdj(nn.Module):
    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n), requires_grad=True)
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            adj = adj.to_dense().to(torch.float32)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

class Adj_renew(nn.Module):
    def __init__(self, node_num, nfeat, nfeat_out, adj_lambda):
        super(Adj_renew, self).__init__()
        self.node_num = node_num
        self.nfeat = nfeat
        self.nfeat_out = nfeat_out
        self.adj_lambda = adj_lambda
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def fit(self, adj, lr):
        estimator = EstimateAdj(adj, symmetric=False, device='cuda').to('cuda').float()
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(), momentum=0.9, lr=lr)
        self.optimizer_l1 = PGD(estimator.parameters(), proxs=[prox_operators.prox_l1], lr=lr, alphas=[5e-4])
        self.optimizer_nuclear = PGD(estimator.parameters(), proxs=[prox_operators.prox_nuclear], lr=lr, alphas=[1.5])

    def the_norm(self):
        return self.estimator._normalize(self.estimator.estimated_adj)

    def forward(self):
        return self.estimator.estimated_adj

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        return torch.trace(XLXT)

    def train_adj(self, features, adj, adv_loss, epoch, lr):
        for param_group in self.optimizer_adj.param_groups:
            param_group["lr"] = lr

        estimator = self.estimator
        estimator.train()
        self.optimizer_adj.zero_grad()

        delta = estimator.estimated_adj - adj
        loss_fro = torch.sum(delta.mul(delta))
        loss_diffiential = 1 * loss_fro + 15 * adv_loss
        loss_diffiential.backward()
        self.optimizer_adj.step()

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        with torch.no_grad():
            estimator.estimated_adj.data.clamp_(0, 1)
            estimator.estimated_adj.data.copy_(
                (estimator.estimated_adj.data + estimator.estimated_adj.data.transpose(0, 1)) / 2
            )

        return estimator.estimated_adj

class EDITS(nn.Module):
    def __init__(self, nfeat, node_num, nfeat_out, adj_lambda, layer_threshold, dropout, lr, weight_decay):
        super(EDITS, self).__init__()
        self.x_debaising = X_debaising(nfeat)
        self.layer_threshold = layer_threshold
        self.adj_renew = Adj_renew(node_num, nfeat, nfeat_out, adj_lambda)
        self.fc = nn.Linear(nfeat * (layer_threshold + 1), 1)
        self.lr = lr

        self.optimizer_feature_l1 = PGD(self.x_debaising.parameters(), proxs=[prox_operators.prox_l1], lr=self.lr, alphas=[5e-6])
        G_params = list(self.x_debaising.parameters())
        self.optimizer_G = optim.RMSprop(G_params, lr=self.lr, eps=1e-04, weight_decay=weight_decay)
        self.optimizer_A = optim.RMSprop(self.fc.parameters(), lr=self.lr, eps=1e-04, weight_decay=weight_decay)
        self.dropout = nn.Dropout(dropout)

    def propagation_cat_new_filter(self, X_de, A_norm, layer_threshold):
        A_norm = A_norm.float()
        X_de = X_de.float()
        X_agg = X_de
        for _ in range(layer_threshold):
            X_de = A_norm.mm(X_de)
            X_agg = torch.cat((X_agg, X_de), dim=1)
        return X_agg

    def forward(self, A, X):
        X = X.float()
        A = A.float()
        X_de = self.x_debaising(X)
        adj_new = self.adj_renew()
        agg_con = self.propagation_cat_new_filter(X_de, adj_new, self.layer_threshold)
        D_pre = self.fc(agg_con)
        D_pre = self.dropout(D_pre)
        return D_pre, adj_new, X_de, agg_con

    def optimize(self, adj, features, idx_train, sens, epoch, lr):
        self.lr = lr
        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizer_A.param_groups:
            param_group["lr"] = lr

        self.train()
        self.optimizer_G.zero_grad()
        self.fc.requires_grad_(False)

        if epoch == 0:
            self.adj_renew.fit(adj, self.lr)

        predictor_sens, _, X_debiased, _ = self.forward(adj, features)
        pos = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
        neg = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)
        adv_loss = - (torch.mean(pos) - torch.mean(neg))
        loss_train = 3e-2 * (X_debiased - features).norm(2) + adv_loss
        loss_train.backward()
        self.optimizer_G.step()
        self.optimizer_feature_l1.zero_grad()
        self.optimizer_feature_l1.step()

        predictor_sens, _, X_debiased, _ = self.forward(adj, features)
        pos = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
        neg = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)
        adv_loss = - (torch.mean(pos) - torch.mean(neg))
        self.adj_renew.train_adj(X_debiased, adj, adv_loss, epoch, lr)

        with torch.no_grad():
            param = self.state_dict()
            param["x_debaising.s"] = torch.clamp(param["x_debaising.s"], 0, 1)
            self.load_state_dict(param)

        for _ in range(8):
            self.fc.requires_grad_(True)
            self.optimizer_A.zero_grad()
            predictor_sens, _, _, _ = self.forward(adj, features)
            pos = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
            neg = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)
            loss_train = torch.abs(torch.mean(pos) - torch.mean(neg))
            loss_train.backward()
            self.optimizer_A.step()
            for p in self.fc.parameters():
                p.data.clamp_(-0.02, 0.02)

        return 0

# MLP
class MLPRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x = data.x  # edge_index 무시
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)

# GCN
class GCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # 단일 회귀 출력

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x).squeeze(-1)  # [N] 형태

# GAT
class GATRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, heads=1):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return self.out(x).squeeze(-1)

# GraphSAGE
class GraphSAGERegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.sage1(x, edge_index))
        x = F.relu(self.sage2(x, edge_index))
        return self.out(x).squeeze(-1)

# GIN
class GINRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gin1 = GINConv(nn1)
        self.gin2 = GINConv(nn2)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gin1(x, edge_index)
        x = self.gin2(x, edge_index)
        return self.out(x).squeeze(-1)

#FnR-GNN
class FnRGNN(nn.Module):
    def __init__(self, nfeat, hidden_dim, dropout, lm, gm, ld, mmd_sample, lr, weight_decay,
                 use_mmd=True, use_gwn=True, use_edge_weight=True):
        super(FnRGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lambda2 = lm
        self.gamma = gm
        self.lambda_dist = ld
        self.mmd_sample_size = mmd_sample

        self.use_mmd = use_mmd
        self.use_gwn = use_gwn
        self.use_edge_weight = use_edge_weight

        self.gcn1 = GCNConv(nfeat, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized") # Sinkhorn 정의 (epsilon = blur 제어)
        # , backend="auto" : Linux에서..

    def compute_mmd(self, h, sensitive_attr):
        # adversarial loss 없이도 직접적인 distance loss로 추가 가능
        # 단점: 해당 속성이 뭔지 명확할 때만 쓸 수 있음. 즉, 민감 속성 라벨이 없는 경우에는 직접 쓰기 힘듦
        # 두 확률 분포 간의 평균 임베딩 차이를 측정해 그룹 간 분포가 비슷해지도록 압력
        # In-processing fairness constraints in model training: 목적: 모델이 학습 중에 민감 그룹 간 차이를 덜 반영하도록 압력 주고 싶다
        # 필요한 이유: fairness constraint를 soft하게 걸 수 있는 방법이 필요할 때, MMD는 differentiable하면서도 샘플 기반이라 유연하고 강력한 도구가 됨.
        mask_0 = (sensitive_attr == 0).squeeze()
        mask_1 = (sensitive_attr == 1).squeeze()
        h_0 = h[mask_0]
        h_1 = h[mask_1]
        
        if h_0.size(0) > self.mmd_sample_size:
            idx_0 = torch.randperm(h_0.size(0), device=h.device)[:self.mmd_sample_size]
            h_0 = h_0[idx_0]
        if h_1.size(0) > self.mmd_sample_size:
            idx_1 = torch.randperm(h_1.size(0), device=h.device)[:self.mmd_sample_size]
            h_1 = h_1[idx_1]
        
        if h_0.numel() == 0 or h_1.numel() == 0:
            return torch.tensor(0.0, device=h.device)
        
        sigma = 1.0
        xx = torch.exp(-torch.cdist(h_0, h_0) / (2 * sigma**2)).mean()
        yy = torch.exp(-torch.cdist(h_1, h_1) / (2 * sigma**2)).mean()
        xy = torch.exp(-torch.cdist(h_0, h_1) / (2 * sigma**2)).mean()
        return xx + yy - 2 * xy

    def compute_edge(self, x, edge_index, sensitive_attr, sim_type='cosine'):
        # 1. 노드 간 특성 유사도 계산
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        if sim_type == 'cosine':
            sim = F.cosine_similarity(x_src, x_dst, dim=1)
        elif sim_type == 'dot':
            sim = torch.sum(x_src * x_dst, dim=1)
        else:
            raise ValueError("Unsupported sim_type. Use 'cosine' or 'dot'.")

        # 2. 민감 속성 차이에 따라 감쇠
        sen_diff = (sensitive_attr[src] != sensitive_attr[dst]).float()

        # 3. 하이브리드 edge 가중치 계산
        edge_weight = sim * torch.exp(-self.gamma * sen_diff)

        # 4. 음수/NaN 방지
        edge_weight = torch.clamp(edge_weight, min=1e-4)

        return edge_weight

    def compute_dist(self, pred, sensitive_attr):
        # 민감 속성 그룹 간 분포가 다르다는 것은 bias가 있을 수 있다는 뜻이고, 이를 fairness 관점에서 줄이려는 것
        # 두 그룹의 representation의 분포를 가깝게
        mask_0 = (sensitive_attr == 0).squeeze()
        mask_1 = (sensitive_attr == 1).squeeze()
        pred_0 = pred[mask_0]
        pred_1 = pred[mask_1]
        mean_diff = torch.abs(pred_0.mean() - pred_1.mean()) if pred_0.numel() > 0 and pred_1.numel() > 0 else 0
        var_diff = torch.abs(pred_0.var() - pred_1.var()) if pred_0.numel() > 0 and pred_1.numel() > 0 else 0
        return mean_diff + var_diff

    def compute_sinkhorn_loss(self, pred, sensitive_attr, sample_size=500):
        pred_0 = pred[sensitive_attr == 0]
        pred_1 = pred[sensitive_attr == 1]
        if pred_0.size(0) == 0 or pred_1.size(0) == 0:
            return torch.tensor(0.0, device=pred.device)
        # Sample to reduce memory
        if pred_0.size(0) > sample_size:
            pred_0 = pred_0[torch.randperm(pred_0.size(0))[:sample_size]]
        if pred_1.size(0) > sample_size:
            pred_1 = pred_1[torch.randperm(pred_1.size(0))[:sample_size]]

        return self.sinkhorn_loss_fn(pred_0, pred_1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        sensitive_attr = data.sensitive_attr

        if self.use_edge_weight:
            edge_weight = self.compute_edge(x, edge_index, sensitive_attr, sim_type='cosine')
        else:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        h = self.gcn1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gcn2(h, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        y = self.classifier(h)

        return y, h

    def optimize(self, data):
        self.train()
        y, h= self.forward(data)
        sensitive_attr =  data.sensitive_attr
        target = data.y.view(-1, 1)

        mse_loss = self.criterion(y, target)  # mse
        mmd_loss = self.compute_mmd(h, sensitive_attr) if self.use_mmd \
        else torch.tensor(0.0, device=h.device, requires_grad=True)
        gwn_loss_1 = self.compute_sinkhorn_loss(y, sensitive_attr) if self.use_gwn \
        else torch.tensor(0.0, device=h.device, requires_grad=True)
        gwn_loss_2 = self.compute_dist(y, sensitive_attr) if self.use_gwn \
        else torch.tensor(0.0, device=h.device, requires_grad=True)
        gwn_loss = gwn_loss_1 + gwn_loss_2
        total_loss = mse_loss + self.lambda2 * mmd_loss + self.lambda_dist * gwn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
                'total_loss': total_loss.item(),
                'mse_loss': mse_loss.item(),
                'mmd_loss': mmd_loss.item(),
                'gwn_loss': gwn_loss.item(),
            }
