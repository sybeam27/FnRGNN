import os
import copy
import torch
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

############################################################################

class German:
    def __init__(self, path):
        if not os.path.exists(path):
            raise ValueError("Data path doesn't exist!")
        else:
            self.data_path = path
        self.raw_data = self._node_process()
        self.A_tensor, self.A = self._edge_process()
        self.senIdx, self.sen_vals, self.trainIdxtensor, self.valIdxTensor, self.testIdxTensor, self.features, self.labels = self._split_data()

    def _node_process(self):
        filenames = os.listdir(self.data_path)

        for file in filenames:
            if os.path.splitext(file)[1] != '.csv':
                continue
            else:
                df_data = pd.read_csv(os.path.join(self.data_path, file))

                # modify str feature
                df_data['GoodCustomer'] = df_data['GoodCustomer'].replace(-1, 0).astype(int)
                gender_map = {'Female': 0, 'Male': 1}
                df_data['Gender'] = df_data['Gender'].map(gender_map).astype(int)

                purposeList = list(df_data['PurposeOfLoan'])
                random.shuffle(purposeList)
                purposeDict = {}
                index = 0
                for pur in purposeList:
                    if purposeDict.get(pur, None) is None:
                        purposeDict[pur] = index
                        index += 1
                    else:
                        continue

                for key in purposeDict.keys():
                    df_data['PurposeOfLoan'] = df_data['PurposeOfLoan'].map(purposeDict).fillna(-1).astype(int)
                    
                return df_data

    def _edge_process(self):
        filenames = os.listdir(self.data_path)

        for file in filenames:
            if os.path.splitext(file)[1] != '.txt':
                continue
            else:
                edges = np.loadtxt(os.path.join(self.data_path, file)).astype(int)

                # Adjacency
                num_dim = len(self.raw_data)
                A = np.zeros((num_dim, num_dim))
                for i in range(len(edges)):
                    A[edges[i][0]][edges[i][1]] = 1
                    A[edges[i][1]][edges[i][0]] = 1
                sym_norm_A = symetric_normalize(A, half=False)
                syn_norm_A_tensor = sp2sptensor(sym_norm_A)
                return syn_norm_A_tensor, A

    def _split_data(self):
        pos_data = self.raw_data[self.raw_data['GoodCustomer']==1]
        pos_index = list(pos_data.index)
        neg_data = self.raw_data[self.raw_data['GoodCustomer']==0]
        neg_index = list(neg_data.index)

        # shuffle the index
        random.seed(20)
        random.shuffle(pos_index)
        random.shuffle(neg_index)

        # split the data
        train_pos_idx = pos_index[:int(0.5*len(pos_index))]
        train_neg_idx = neg_index[:int(0.5*len(neg_index))]
        val_pos_idx = pos_index[int(0.5*len(pos_index)): int(0.75*len(pos_index))]
        val_neg_idx = neg_index[int(0.5*len(neg_index)): int(0.75*len(neg_index))]
        test_pos_idx = pos_index[int(0.75*len(pos_index)):]
        test_neg_idx = neg_index[int(0.75*len(neg_index)):]

        trainIdx = train_pos_idx + train_neg_idx
        random.shuffle(trainIdx)
        valIdx = val_pos_idx + val_neg_idx
        random.shuffle(valIdx)
        testIdx = test_pos_idx + test_neg_idx
        random.shuffle(testIdx)

        assert len(trainIdx)+len(valIdx)+len(testIdx) == len(self.raw_data), "Missing data or leaking data!"

        feature_cols = list(self.raw_data.columns)
        feature_cols.remove('GoodCustomer')
        sen_idx = feature_cols.index('Gender')
        sen_vals = self.raw_data['Gender'].values.astype(int)
        feature_data = self.raw_data[feature_cols]
        labels = self.raw_data['GoodCustomer']

        # transform to tensor
        trainIdxTensor = torch.LongTensor(trainIdx)
        valIdxTensor = torch.LongTensor(valIdx)
        testIdxTensor = torch.LongTensor(testIdx)
        featuredata = torch.FloatTensor(np.array(feature_data))
        labels = torch.LongTensor(np.array(labels))

        return sen_idx, sen_vals, trainIdxTensor, valIdxTensor, testIdxTensor, featuredata, labels

    def get_index(self):
        return [self.trainIdxtensor, self.valIdxTensor, self.testIdxTensor]

    def get_raw_data(self):
        return [self.features, self.A_tensor, self.labels]

    def generate_counterfactual_perturbation(self, data):
        feature_data = copy.deepcopy(data)
        feature_data[:, self.senIdx] = 1 - feature_data[:, self.senIdx]
        return feature_data

    def generate_node_perturbation(self, prob: float, sen: bool = False):
        feature_data = copy.deepcopy(self.features)
        r = np.random.binomial(n=1, p=prob, size=feature_data.numpy().shape)
        for i in range(len(feature_data)):
            r[i][self.senIdx] = 0
        noise = np.multiply(r, np.random.normal(0., 1., r.shape))
        noise_tensor = torch.FloatTensor(noise)
        x_hat = feature_data + noise_tensor

        if sen:
            x_hat = self.generate_counterfactual_perturbation(x_hat)
        return x_hat

    def generate_struc_perturbation(self, drop_prob: float, tensor: bool = True):
        A = copy.deepcopy(self.A)
        half_A = np.triu(A)
        row, col = np.nonzero(half_A)
        idx_perturb = np.random.binomial(n=1, p=1-drop_prob, size=row.shape)
        broken_edges = np.where(idx_perturb==0)[0]
        for idx in broken_edges:
            half_A[row[idx]][col[idx]] = 0
        new_A = symetric_normalize(half_A, half=True)
        if tensor:
            new_A = sp2sptensor(new_A)
        return new_A

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def sp2sptensor(m):
    sparse_m = sp.coo_matrix(m).astype(np.float64)
    indices = torch.from_numpy(np.vstack((sparse_m.row, sparse_m.col))).long()
    values = torch.from_numpy(sparse_m.data)
    shape = torch.Size(sparse_m.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def is_symmetric(m):
    res = np.int64(np.triu(m).T == np.tril(m))
    if np.where(res == 0)[0].size > 0:
        raise ValueError("The matrix is not symmetric!")
    else:
        pass

def symetric_normalize(m, half: bool):
    if not half:
        is_symmetric(m)
    else:
        m = m + m.T - np.diag(np.diagonal(m))

    hat_m = m + np.eye(m.shape[0])
    D = np.sum(hat_m, axis=1)
    D = np.diag(D)

    with np.errstate(divide='ignore'):
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0

    D[np.isinf(D)] = 0
    sn_m = np.matmul(np.matmul(D, hat_m), D)
    return sn_m

def dataset_config():
    config = {
        'region_job_r': {
            'path': "./dataset/pokec",
            'dataset': 'region_job', # Pokec_z
            'predict_attr': 'completion_percentage',  # completion_percentage: 프로필을 얼마나 채웠는가 (%). 자발적 행동의 간접 지표.
            'sens_attr': 'region', # gender
            'label_number': 500,
            'sens_number': 200,
            'test_idx': False,
            'dn': 'Pokec_z_Region'
        },

        'region_job_g': {
            'path': "./dataset/pokec",
            'dataset': 'region_job',
            'predict_attr': 'completion_percentage',  # completion_percentage: 프로필을 얼마나 채웠는가 (%). 자발적 행동의 간접 지표.
            'sens_attr': 'gender', # gender
            'label_number': 500,
            'sens_number': 200,
            'test_idx': False,
            'dn': 'Pokec_z_Gender'
        },

        'region_job_2_r': {
            'path': "./dataset/pokec",
            'dataset': 'region_job_2',
            'predict_attr': 'completion_percentage',  # completion_percentage: 프로필을 얼마나 채웠는가 (%). 자발적 행동의 간접 지표.
            'sens_attr': 'region', # gender
            'label_number': 500,
            'sens_number': 200,
            'test_idx': False,
            'dn': 'Pokec_n_Region'
        },

        'region_job_2_g': {
            'path': "./dataset/pokec",
            'dataset': 'region_job_2',
            'predict_attr': 'completion_percentage',  # completion_percentage: 프로필을 얼마나 채웠는가 (%). 자발적 행동의 간접 지표.
            'sens_attr': 'gender', # gender
            'label_number': 500,
            'sens_number': 200,
            'test_idx': False,
            'dn': 'Pokec_n_Gender'
        },

        'nba_p': {
            'path': "./dataset/NBA",
            'dataset': 'nba',
            # PIE (Player Impact Estimate): 통합 퍼포먼스 지표, 퍼포먼스에 대한 차별 여부 확인 가능
            # MPG (Minutes Per Game): 출전 시간 → 팀의 코치 결정, 제도적 편향 가능성
            'predict_attr': 'PIE',
            'sens_attr': 'country',  # AGE, palyer_height, player_weight
            'label_number': 100,
            'sens_number': 50,
            'test_idx': True,
            'dn': 'NBA(PIE)_Country'
        },

        'nba_m': {
            'path': "./dataset/NBA",
            'dataset': 'nba',
            'predict_attr': 'MPG',
            'sens_attr': 'country',
            'label_number': 100,
            'sens_number': 50,
            'test_idx': True,
            'dn': 'NBA(MPG)_Country'
        },

        'german_g': {
            'path': './dataset/NIFTY',
            'dataset': 'german',
            'predict_attr': 'LoanAmount', # LoanAmount(대출 금액), LoanRateAsPercentOfIncome(소득대비상환비율), YearsAtCurrentHome(거주연수)
            'sens_attr': 'Gender', # Gender, ForeignWorker, Single(독신여부), HasTelephone(전화기보유여부), OwnsHouse(주택소유여부), Unemployed(실직상태여부) Age(고령자로 나눠서 가능) 등등등..
            'label_number': None,
            'sens_number': None,
            'test_idx': None,
            'dn': 'German_Gender'
        },

        'german_f': {
            'path': './dataset/NIFTY',
            'dataset': 'german',
            'predict_attr': 'LoanAmount',
            'sens_attr': 'ForeignWorker',
            'label_number': None,
            'sens_number': None,
            'test_idx': None,
            'dn': 'German_ForeignWorker'
        },

        'german_s': {
            'path': './dataset/NIFTY',
            'dataset': 'german',
            'predict_attr': 'LoanAmount', 
            'sens_attr': 'Single', 
            'label_number': None,
            'sens_number': None,
            'test_idx': None,
            'dn': 'German_Single'
        },

        'german_t': {
            'path': './dataset/NIFTY',
            'dataset': 'german',
            'predict_attr': 'LoanAmount', 
            'sens_attr': 'HasTelephone', 
            'label_number': None,
            'sens_number': None,
            'test_idx': None,
            'dn': 'German_HasTelephone'
        },

        'german_h': {
            'path': './dataset/NIFTY',
            'dataset': 'german',
            'predict_attr': 'LoanAmount', 
            'sens_attr': 'OwnsHouse', 
            'label_number': None,
            'sens_number': None,
            'test_idx': None,
            'dn': 'German_OwnsHouse'
        },

        'german_e': {
            'path': './dataset/NIFTY',
            'dataset': 'german',
            'predict_attr': 'LoanAmount', 
            'sens_attr': 'Unemployed', 
            'label_number': None,
            'sens_number': None,
            'test_idx': None,
            'dn': 'German_Unemployed'
        }
    }
    return config

def load_dataset(dataset, sens_attr,predict_attr, seed, path, sens_number):
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt(os.path.join(path, f"{dataset}_relationship.txt"), dtype=np.int64)

    mapped_edges = []
    dropped = 0
    for u, v in edges_unordered:
        u_mapped = idx_map.get(u)
        v_mapped = idx_map.get(v)
        if u_mapped is not None and v_mapped is not None:
            mapped_edges.append((u_mapped, v_mapped))
        else:
            dropped += 1

    print(f"[INFO] 유효하지 않은 user_id로 인해 제거된 edge 수: {dropped}")
    edges = np.array(mapped_edges, dtype=int)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]

    # random.shuffle(label_idx)
    # idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    # idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    # if test_idx:
    #     idx_test = label_idx[label_number:]
    #     idx_val = idx_test
    # else:
    #     idx_test = label_idx[int(0.75 * len(label_idx)):]

    # sens = idx_features_labels[sens_attr].values
    # sens_idx = set(np.where(sens >= 0)[0])
    # idx_test = np.asarray(list(sens_idx & set(idx_test)))
    # sens = torch.FloatTensor(sens)
    # idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    # random.seed(seed)
    # random.shuffle(idx_sens_train)
    # idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # stratified sampling
    sens_all = idx_features_labels[sens_attr].values
    labels_all = labels  # np.ndarray 혹은 torch.Tensor

    valid_idx = np.where((labels_all >= 0) & (sens_all >= 0))[0]
    sens_valid = sens_all[valid_idx]
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    train_part, temp_part = next(sss1.split(valid_idx, sens_valid))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_part, test_part = next(sss2.split(temp_part, sens_valid[temp_part]))

    idx_train = torch.LongTensor(valid_idx[train_part])
    idx_val = torch.LongTensor(valid_idx[temp_part[val_part]])
    idx_test = torch.LongTensor(valid_idx[temp_part[test_part]])

    sens = torch.FloatTensor(sens_all)
    sens_idx = set(np.where(sens_all >= 0)[0])
    idx_sens_train_pool = list(sens_idx - set(idx_val.numpy()) - set(idx_test.numpy()))
    random.seed(seed)
    random.shuffle(idx_sens_train_pool)
    idx_sens_train = torch.LongTensor(idx_sens_train_pool[:sens_number])
    
    # random.shuffle(sens_idx)

    if dataset == 'nba':
        features = feature_norm(features)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train
  
def load_dataset_unified(dataset, sens_attr, predict_attr, seed, path="./dataset/", sens_number=500, test_idx=False):
    
    if dataset.lower() != 'german':
        full_path = path
        return load_dataset(dataset, sens_attr, predict_attr, seed, full_path, sens_number)

    elif dataset.lower() == 'german':
        full_path = path
        print('Loading {} dataset from {}'.format(dataset, full_path))
        german_data = German(full_path)
        
        adj = german_data.A_tensor
        features = german_data.features
        labels = german_data.labels
        sens = torch.FloatTensor(german_data.sen_vals)
        # idx_train, idx_val, idx_test = german_data.get_index()

        # # 민감 속성 학습용 인덱스 (val/test 제외)
        # all_idx = set(range(len(sens)))
        # sens_idx = set(np.where(sens.numpy() >= 0)[0])
        # idx_sens_train = list(sens_idx - set(idx_val.numpy()) - set(idx_test.numpy()))
        # random.seed(seed)
        # random.shuffle(idx_sens_train)
        # idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        # Stratified split: 50/25/25
        total_idx = np.where((labels >= 0) & (sens >= 0))[0]
        sens_np = sens[total_idx].numpy()

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        train_idx, temp_idx = next(sss1.split(total_idx, sens_np))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_idx, test_idx = next(sss2.split(temp_idx, sens_np[temp_idx]))

        idx_train = torch.LongTensor(total_idx[train_idx])
        idx_val = torch.LongTensor(total_idx[temp_idx[val_idx]])
        idx_test = torch.LongTensor(total_idx[temp_idx[test_idx]])

        sens_idx = set(np.where(sens.numpy() >= 0)[0])
        idx_sens_train = list(sens_idx - set(idx_val.numpy()) - set(idx_test.numpy()))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def load_and_prepare_dataset(dataset_config, config_name, seed, normalize=True):
    cfg = dataset_config.get(config_name)
    
    if cfg is None:
        raise ValueError(f"Dataset '{config_name}' not found in dataset_config.")
    
    path = cfg['path']
    dataset_name = cfg['dataset']
    predict_attr = cfg['predict_attr']
    sens_attr = cfg['sens_attr']
    label_number = cfg.get('label_number')
    sens_number = cfg.get('sens_number')
    test_idx = cfg.get('test_idx')

    df = pd.read_csv(f"{path}/{dataset_name}.csv")

    # Load graph data
    adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_dataset_unified(
        dataset_name, sens_attr, predict_attr, seed, path, sens_number, test_idx
    )

    # Remove disconnected nodes (degree == 0)
    if isinstance(adj, torch.Tensor):
        deg = torch.sparse.sum(adj, dim=1).to_dense().numpy()
    else:
        deg = np.array(adj.sum(axis=1)).flatten()

    connected_mask = deg > 0
    connected_idx = np.where(connected_mask)[0]

    # Make new index mapping: old → new
    old_to_new = {old: new for new, old in enumerate(connected_idx)}

    # Filter node-level attributes
    features = features[connected_idx]
    labels = labels[connected_idx]
    sens = sens[connected_idx]

    # Re-map indices
    idx_train = torch.tensor([old_to_new[i.item()] for i in idx_train if i.item() in old_to_new])
    idx_val = torch.tensor([old_to_new[i.item()] for i in idx_val if i.item() in old_to_new])
    idx_test = torch.tensor([old_to_new[i.item()] for i in idx_test if i.item() in old_to_new])
    idx_sens_train = torch.tensor([old_to_new[i.item()] for i in idx_sens_train if i.item() in old_to_new])

    # Filter edge list
    if isinstance(adj, torch.Tensor):
        edge_index = adj._indices().numpy()
        src, dst = edge_index[0], edge_index[1]
    else:
        adj = adj.tocoo()
        src, dst = adj.row, adj.col

    mask = np.isin(src, connected_idx) & np.isin(dst, connected_idx)
    src_filtered = [old_to_new[i] for i in src[mask]]
    dst_filtered = [old_to_new[i] for i in dst[mask]]
    edge_index = torch.tensor([src_filtered, dst_filtered], dtype=torch.long)

    # Rebuild sparse adj
    N = len(connected_idx)
    values = np.ones(len(src_filtered))
    new_adj = sp.coo_matrix((values, (src_filtered, dst_filtered)), shape=(N, N))
    new_adj = torch.sparse_coo_tensor(
        indices=torch.tensor([src_filtered, dst_filtered]),
        values=torch.ones(len(src_filtered)),
        size=(N, N)
    )

    # Binary sensitive attribute
    if sens_attr:
        sens[sens > 0] = 1
    print(f"[{dataset_name}] sens=0: {torch.sum(sens == 0).item()}, sens=1: {torch.sum(sens == 1).item()}")
    print('-' * 50)

    # Normalize features and labels
    x_scaler, y_scaler = None, None
    if normalize:
        x_scaler = StandardScaler()
        features = torch.tensor(x_scaler.fit_transform(features.cpu()), dtype=torch.float32)

        y_scaler = StandardScaler()
        labels = torch.tensor(y_scaler.fit_transform(labels.cpu().reshape(-1, 1)), dtype=torch.float32).view(-1)

    # Build PyG Data object
    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels.float(),
        sensitive_attr=sens,
        adj=new_adj  # now filtered adj
    )
    data.idx_train = idx_train
    data.idx_val = idx_val
    data.idx_test = idx_test
    data.idx_sens_train = idx_sens_train

    return data, df, cfg

def print_sensitive_attr_distribution(data_dict):
    sens = data_dict['sensitive_attr']  # torch.FloatTensor
    for split in ['idx_train', 'idx_val', 'idx_test']:
        idx = data_dict[split]
        values = sens[idx].numpy()
        unique, counts = np.unique(values, return_counts=True)
        total = len(values)
        print(f"[{split}] 총 {total}개")
        for u, c in zip(unique, counts):
            print(f"  민감속성 {int(u)}: {c}개 ({c/total:.1%})")
        print("-" * 30)
