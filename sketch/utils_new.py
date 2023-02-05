import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import datetime

import sys
import pickle as pkl
import scipy.io as sio
import networkx as nx
from collections import defaultdict

from copy import deepcopy

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from networkx.readwrite import json_graph
import json
import pandas as pd

from collections import defaultdict
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import networkx as nx
import time
import sys
import os

# import multiprocessing as mp

"""
Load data function adopted from https://github.com/williamleif/GraphSAGE
"""
WALK_LEN = 5
N_WALKS = 50

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data_graphsage(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        def conversion(n): return int(n)
    else:
        def conversion(n): return n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        def lab_conversion(n): return n
    else:
        def lab_conversion(n): return int(n)

    class_map = {conversion(k): lab_conversion(v)
                 for k, v in class_map.items()}

    # Remove all nodes that do not have val/test annotations
    # (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G._node[node] or not 'test' in G._node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(
        broken_count))

    # Make sure the graph has edge train_removed annotations
    # (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G._node[edge[0]]['val'] or G._node[edge[1]]['val'] or
                G._node[edge[0]]['test'] or G._node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[str(n)] for n in G.nodes(
        ) if not G._node[n]['val'] and not G._node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

"""
Load data function adopted from https://github.com/tkipf/gcn
"""

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_gcn(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        # with open("data/gcn/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file(
    #     "data/gcn/ind.{}.test.index".format(dataset_str))
    test_idx_reorder = parse_index_file(
        "./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    G = nx.from_dict_of_lists(graph)

    edges = []
    for s in G:
        for t in G[s]:
            if s!=t:
                edges += [[s, t]]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels.argmax(axis=1)  # pytorch require target 1d

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))

    edges = np.array(edges)
    adj_matrix = get_adj(edges, features.shape[0])
    
    return adj_matrix, np.array(labels), features.toarray(),\
        np.array(idx_train), np.array(idx_val), np.array(idx_test)


def preprocess_data(dataset):
    if dataset in ['ppi', 'ppi-large', 'reddit', 'flickr', 'yelp']:
        prefix = './data/{}/{}'.format(dataset, dataset)
        G, feats, id_map, walks, class_map = load_data_graphsage(prefix)

        degrees = np.zeros(len(G), dtype=np.int64)
        edges = []
        labels = []
        idx_train = []
        idx_val = []
        idx_test = []
        for s in G:
            if G.nodes[s]['test']:
                idx_test += [s]
            elif G.nodes[s]['val']:
                idx_val += [s]
            else:
                idx_train += [s]
            for t in G[s]:
                if s!=t:
                    edges += [[s, t]]
            degrees[s] = len(G[s])
            labels += [class_map[str(s)]]

        edges = np.array(edges)
        adj_matrix = get_adj(edges, feats.shape[0])

        return adj_matrix, np.array(labels), np.array(feats), \
            np.array(idx_train), np.array(idx_val), np.array(idx_test)

    elif dataset in ['cora', 'citeseer', 'pubmed']:
        # dataset=='cora' or dataset=='citeseer' or dataset=='pubmed':
        return load_data_gcn(dataset)

def sym_normalize(mx):
    """Sym-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    
    colsum = np.array(mx.sum(0))
    c_inv = np.power(colsum, -1/2).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    
    mx = r_mat_inv.dot(mx).dot(c_mat_inv)
    return mx

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


def generate_random_graph(n, e, prob = 0.1):
    idx = np.random.randint(2)
    g = nx.powerlaw_cluster_graph(n, e, prob) 
    adj_lists = defaultdict(set)
    num_feats = 8
    degrees = np.zeros(len(g), dtype=np.int64)
    edges = []
    for s in g:
        for t in g[s]:
            edges += [[s, t]]
            degrees[s] += 1
            degrees[t] += 1
    edges = np.array(edges)
    return degrees, edges, g, None 

def get_sparse(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj) 

def norm(l):
    return (l - np.average(l)) / np.std(l)

def stat(l):
    return np.average(l), np.sqrt(np.var(l))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape
    
def sparse_mx_to_torch_csr_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape


# def get_adj(edges, num_nodes):
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                     shape=(num_nodes, num_nodes), dtype=np.float32)
#     return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

def get_adj(edges, num_nodes):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    assert(adj.diagonal().sum()==0)
    assert((adj!=adj.T).nnz==0 )
    # print(adj.diagonal().sum())
    return adj


def normalize_lap(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj

def normalize_lap_2(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv      = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = (adj - sp.eye(adj.shape[0])).dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj, sp.diags(d_inv, 0)

# def get_laplacian(adj):
#     # normalize tilde_A = A + I by 
#     # D^{-1/2} tilde_A D^{-1/2}
#     adj = normalize_lap(adj + sp.eye(adj.shape[0]))
#     return sparse_mx_to_torch_sparse_tensor(adj) 


# Transfer matrix list to device
def package_mxl(mxl, device):
    # FloatTensor 转化成 coo matrix，这个API已经不用了
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]

def package_mxl_csr(mxl, device):
    temp = mxl[0]
    print(temp)
    return [torch._sparse_csr_tensor(torch.from_numpy(mx[0]), 
        torch.from_numpy(mx[1]), torch.from_numpy(mx[2]), 
        size=mx[3]).to(device) for mx in mxl]

# add loss record
def record_result_new(args, txt_name, total_time_all, samp_num_list, valid_f1_all, 
                  valid_loss_all, test_f1_all, epoch_num, epoch_time_all, write_file, 
                  sample_method, original_stdout):
    dir_name = './result/{}'.format(args.dataset)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    with open('./result/{}/{}'.format(args.dataset, txt_name), 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        np.set_printoptions(precision=5)
        print(args)
        print("{}_repeat {} times".format(args.dataset, args.n_trial))
        print("batch_size: {}, base_sample_num: {}, layers: {}".format(args.batch_size, 
              args.samp_num, args.n_layers))
        print("samp_num_each_layer", samp_num_list)
        print("-" * 20)

        print("Sampler method: ", sample_method)
        f1_mean, f1_mean_sd = np.average(test_f1_all), np.std(test_f1_all) / np.sqrt(args.n_trial)
        epoch_mean, epoch_sd = np.mean(epoch_num), np.std(epoch_num) / np.sqrt(args.n_trial)
        print("f1.mean", "f1.se")
        print(np.array([f1_mean, f1_mean_sd]))
        print("f1 95% CI")
        print(np.array([f1_mean - 1.96 * f1_mean_sd, f1_mean + 1.96 * f1_mean_sd]))
        print("epoch_mean, epoch_mean_sd")
        print([epoch_mean, epoch_sd])
        print("training time: mean, mean's sd")
        print(np.array([np.mean(total_time_all), np.std(total_time_all) / np.sqrt(args.n_trial)]))
        print("\n")
        print("_" * 20)
    
    sys.stdout = original_stdout # Reset the standard output to its original value

    # record the data to .pkl

    cur_result = dict()
    cur_result["args"] = args
    cur_result["test_f1"] = test_f1_all
    cur_result["f1 mean, mean sd"] = [f1_mean, f1_mean_sd]
    cur_result["time"] = total_time_all
    cur_result["avg time, avg std"] = [np.mean(total_time_all), np.std(total_time_all) / np.sqrt(args.n_trial)]
    cur_result["epoch_time_all"] = epoch_time_all
    cur_result["epoch_num"] = epoch_num
    cur_result["epoch mean, meand= sd"] = [epoch_mean, epoch_sd]
    cur_result["valid_f1_all"] = valid_f1_all
    cur_result["valid_loss_all"] = valid_loss_all
    cur_result["layer_samp_num"] = samp_num_list
    return cur_result

def get_test_metric(best_model, batch_nodes, feat_data, 
    full_sampler, lap_matrix, depth, device):
    
    adjs, input_nodes, output_nodes, _ = full_sampler(np.random.randint(2**32 - 1), batch_nodes, None, None, lap_matrix, depth)
    adjs = package_mxl(adjs, device)
    output = best_model.forward(feat_data[input_nodes], adjs)
    
    print(output.shape, type(output))
    
    return output
    
def gpu_warmup(device):
    a = torch.ones(10000, device=device)
    for i in range(100): a = a + a
    
    del a
    torch.cuda.empty_cache()

def estWRS_weights(p, m):
    n = len(p)
    wrs_index = np.random.choice(n, m, False, p)

    weights = np.zeros(m)
    p_sum = 0
    
    for i in range(m):
        
        alpha = n / (i + 1) / (n - i)
        weights[i] = (1-p_sum) / p[wrs_index[i]] * alpha
        weights[:i] = weights[:i] * (1 - alpha) + alpha
        p_sum += p[wrs_index[i]]

    return wrs_index, weights
    