# import torch
# import pandas as pd
# import pdb
# import argparse
# import pickle
# import os
# import numpy as np
# import random
# import tqdm
# import time
# from TAR import *
# from my import *
import pandas as pd
import numpy as np
import pickle
import pdb
import time
import tqdm
import random
def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

K_train = 10000
K_test = 2000
root = '../../data/DBpedia/mid/'
kg_dict_all = load_obj(root + 'kg_dict_all.pkl')
is_dict_all = load_obj(root + 'is_dict_all.pkl')
kg_dict_train = load_obj(root + 'kg_dict_train.pkl')
is_dict_train = load_obj(root + 'is_dict_train.pkl')
kg_data_all = pd.read_csv(root + 'kg_data_all.csv', index_col=0)
is_data_all = pd.read_csv(root + 'is_data_all.csv', index_col=0)
kg_data_train = pd.read_csv(root + 'kg_data_train.csv', index_col=0)
is_data_train = pd.read_csv(root + 'is_data_train.csv', index_col=0)
ot_data = pd.read_csv(root + 'ot.csv', index_col=0)
save_root = '../../data/DBpedia/input/'
def get_h_2_rt(dict_1):
    dict_2 = {}
    for key in dict_1:
        es = dict_1[key]
        for e in es:
            try:
                dict_2[key[0]].add((key[1], e))
            except:
                dict_2[key[0]] = set([(key[1], e)])
    return dict_2
def get_t_2_hr(dict_1):
    dict_2 = {}
    for key in dict_1:
        es = dict_1[key]
        for e in es:
            try:
                dict_2[e].add((key[0], key[1]))
            except:
                dict_2[e] = set([(key[0], key[1])])
    return dict_2
def get_train_answers_2p_e(train_answers_1p, k=K_train):
    ret = {}
    h_2_rt = get_h_2_rt(train_answers_1p)
    while len(ret) < k:
        query_1 = random.choice(list(train_answers_1p.keys()))
        answers_1 = list(train_answers_1p[query_1])
        answer_1 = random.choice(answers_1)
        try:
            queries_2 = list(h_2_rt[answer_1])
            query_2 = random.choice(queries_2)
            if len(set([query_1[0], answer_1, query_2[1]])) != 3:
                continue
        except:
            continue
        ret[(query_1[0], query_1[1], query_2[0])] = query_2[1]
    return ret

def get_train_answers_2p_c(train_answers_1p, is_dict, k=K_train):
    ret = {}
    h_2_rt = get_h_2_rt(train_answers_1p)
    while len(ret) < k:
        query_1 = random.choice(list(train_answers_1p.keys()))
        answers_1 = list(train_answers_1p[query_1])
        answer_1 = random.choice(answers_1)
        try:
            queries_2 = list(h_2_rt[answer_1])
            query_2 = random.choice(queries_2)
            if len(set([query_1[0], answer_1, query_2[1]])) != 3:
                continue
            concepts = list(is_dict[query_2[1]])
            concept = random.choice(concepts)
        except:
            continue
        ret[(query_1[0], query_1[1], query_2[0])] = concept
    return ret

def get_test_answers_2p_e(train_answers_1p, test_answers_1p, k=K_test):
    ret = {}
    h_2_rt = get_h_2_rt(test_answers_1p)
    while len(ret) < k:
        query_1 = random.choice(list(train_answers_1p.keys()))
        answers_1 = list(train_answers_1p[query_1])
        answer_1 = random.choice(answers_1)
        try:
            queries_2 = list(h_2_rt[answer_1])
            query_2 = random.choice(queries_2)
            if len(set([query_1[0], answer_1, query_2[1]])) != 3:
                continue
        except:
            continue
        ret[(query_1[0], query_1[1], query_2[0])] = query_2[1]
    return ret

def get_test_answers_2p_c(train_answers_1p, test_answers_1p, is_dict, k=K_test):
    ret = {}
    h_2_rt = get_h_2_rt(test_answers_1p)
    while len(ret) < k:
        query_1 = random.choice(list(train_answers_1p.keys()))
        answers_1 = list(train_answers_1p[query_1])
        answer_1 = random.choice(answers_1)
        try:
            queries_2 = list(h_2_rt[answer_1])
            query_2 = random.choice(queries_2)
            if len(set([query_1[0], answer_1, query_2[1]])) != 3:
                continue
            concepts = list(is_dict[query_2[1]])
            concept = random.choice(concepts)
        except:
            continue
        ret[(query_1[0], query_1[1], query_2[0])] = concept
    return ret

def get_train_filter_e_2p(train_answers_1p_e, train_answers_2p_e):
    ret = {}
    for query in train_answers_2p_e:
        filters = set()
        answers_1 = train_answers_1p_e.get((query[0], query[1]), set())
        for answer_1 in answers_1:
            answers_2 = train_answers_1p_e.get((answer_1, query[2]), set())
            for answer_2 in answers_2:
                filters.add(answer_2)
        ret[query] = filters
    return ret

def get_train_filter_c_2p(train_answers_1p_e, train_answers_2p_c, is_dict_train):
    ret = {}
    for query in train_answers_2p_c:
        filters = set()
        answers_1 = train_answers_1p_e.get((query[0], query[1]), set())
        for answer_1 in answers_1:
            answers_2 = train_answers_1p_e.get((answer_1, query[2]), set())
            for answer_2 in answers_2:
                concepts = is_dict_train.get(answer_2, set())
                for concept in concepts:
                    filters.add(concept)
        ret[query] = filters
    return ret

def get_test_filter_e_2p(train_answers_1p_e, test_answers_2p_e):
    ret = {}
    for query_test in test_answers_2p_e:
        filters = set()
        answers_1 = train_answers_1p_e.get((query_test[0], query_test[1]), set())
        for answer_1 in answers_1:
            answers_2 = train_answers_1p_e.get((answer_1, query_test[2]), set())
            for answer_2 in answers_2:
                filters.add(answer_2)
        ret[query_test] = filters
    return ret

def get_test_filter_c_2p(train_answers_1p_e, test_answers_2p_c, is_dict_train):
    ret = {}
    for query_test in test_answers_2p_c:
        filters = set()
        answers_1 = train_answers_1p_e.get((query_test[0], query_test[1]), set())
        for answer_1 in answers_1:
            answers_2 = train_answers_1p_e.get((answer_1, query_test[2]), set())
            for answer_2 in answers_2:
                concepts = is_dict_train.get(answer_2, set())
                for concept in concepts:
                    filters.add(concept)
        ret[query_test] = filters
    return ret



# train_answers_2p_e = get_train_answers_2p_e(train_answers_1p_e)
# train_answers_2p_c = get_train_answers_2p_c(train_answers_1p_e, is_dict_train)
# test_answers_2p_e = get_test_answers_2p_e(train_answers_1p_e, test_answers_1p_e)
# test_answers_2p_c = get_test_answers_2p_c(train_answers_1p_e, test_answers_1p_e, is_dict_all)
# train_filter_answers_2p_e = get_train_filter_e_2p(train_answers_1p_e, train_answers_2p_e)
# train_filter_answers_2p_c = get_train_filter_c_2p(train_answers_1p_e, train_answers_2p_c, is_dict_train)
# test_filter_answers_2p_e = get_test_filter_e_2p(train_answers_1p_e, test_answers_2p_e)
# test_filter_answers_2p_c = get_test_filter_c_2p(train_answers_1p_e, test_answers_2p_c, is_dict_train)

# ret_2p = {'train': {'e': train_answers_2p_e, 'c': train_answers_2p_c},
#           'test': {'e': test_answers_2p_e, 'c': test_answers_2p_c},
#           'train_filter': {'e': train_filter_answers_2p_e, 'c': train_filter_answers_2p_c},
#           'test_filter': {'e': test_filter_answers_2p_e, 'c': test_filter_answers_2p_c}}
# save_obj(ret_2p, save_root + '2p.pkl')
# print('Done 2p')



data_set = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('B', 'E'),
    ('C', 'F'),
    ('E', 'G'),
    # 在这里添加更多的父子关系数据
]

tree = {}

for parent, child in data_set:
    if parent not in tree:
        tree[parent] = []
    tree[parent].append(child)

def build_tree(tree, root):
    if root not in tree:
        return {}
    children = {}
    for child in tree[root]:
        children[child] = build_tree(tree, child)
    return children

root_node = 'A'
multi_tree = {root_node: build_tree(tree, root_node)}

print(multi_tree)

tree = {}

for parent, child in data_set:
    if parent not in tree:
        tree[parent] = []
    tree[parent].append(child)


def build_tree(tree, root):
    if root not in tree:
        return {}
    children = {}
    for child in tree[root]:
        children[child] = build_tree(tree, child)
    return children


def find_paths(node, path, result):
    # 将当前节点加入路径
    path.append(node)

    # 如果是叶子节点，将路径加入结果集
    if not node:
        result.append(path.copy())
    else:
        # 递归处理子节点
        for child in node.keys():
            find_paths(node[child], path, result)

    # 移除当前节点，回溯到上一层
    path.pop()


root_node = 'A'
multi_tree = {root_node: build_tree(tree, root_node)}

# 初始化结果集
result = []

# 调用函数开始遍历
find_paths(multi_tree[root_node], [], result)

# 输出结果集
for path in result:
    print(path)

# save_root = '../tmp/'
# cfg = parse_args(["--dataset","DBpedia"])
# seed_everything(cfg.seed)
# device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
# e_dict, c_dict, r_dict, ot, is_data_train = get_mapper(cfg.root + cfg.dataset + '/')
#
# model = TAR(cfg.emb_dim, e_dict, c_dict, r_dict)
# model_path = save_root + str(900)
#
# model.load_state_dict(torch.load(model_path))
# train_e_1p, train_c_1p, train_filter_e_1p, train_filter_c_1p, test_e_1p, test_c_1p, test_filter_e_1p, test_filter_c_1p, valid_dataloader_1p_e, valid_dataloader_1p_c, test_dataloader_1p_e, test_dataloader_1p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='1p')
# train_e_2p, train_c_2p, train_filter_e_2p, train_filter_c_2p, test_e_2p, test_c_2p, test_filter_e_2p, test_filter_c_2p, valid_dataloader_2p_e, valid_dataloader_2p_c, test_dataloader_2p_e, test_dataloader_2p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2p')
# train_e_3p, train_c_3p, train_filter_e_3p, train_filter_c_3p, test_e_3p, test_c_3p, test_filter_e_3p, test_filter_c_3p, valid_dataloader_3p_e, valid_dataloader_3p_c, test_dataloader_3p_e, test_dataloader_3p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3p')
#
# indices = my_evaluate(model, get_dict_gt(e_dict, r_dict, '2p'), valid_dataloader_2p_e, 'cpu', '2p')
# print(indices)

# import networkx as nx
# import random
#
# def find_all_paths(graph, start, end, n):
#     paths = []
#     visited = set()
#
#     def dfs(node, path):
#         print(path)
#         if len(path) == n+1:
#             if node == end:
#                 paths.append(path)
#             return
#
#         visited.add(node)
#         for neighbor in graph.neighbors(node):
#             if neighbor not in visited:
#                 dfs(neighbor, path + [neighbor])
#         visited.remove(node)
#
#     dfs(start, [start])
#     return paths
#
#
# # 创建一个空的有向图
# G = nx.DiGraph()
#
# # 添加节点
# num_nodes = 50
# nodes = range(num_nodes)
# G.add_nodes_from(nodes)
#
# # 每个节点的度数
# degree = 5
#
# # 添加有向边
# for node in nodes:
#     # 获取当前节点的出边邻居节点
#     out_neighbors = list(G.successors(node))
#
#     # 随机选择未连接的出边邻居节点
#     non_neighbors = list(set(nodes) - set(out_neighbors) - {node})
#     random.shuffle(non_neighbors)
#
#     # 添加出边，直到节点的出度达到要求
#     for i in range(degree - len(out_neighbors)):
#         out_neighbor_node = non_neighbors[i]
#         G.add_edge(node, out_neighbor_node)
#
# # 打印图的节点数和边数
# print("节点数：", G.number_of_nodes())
# print("边数：", G.number_of_edges())
#
# # 寻找路径
# start_node = 1
# end_node = 7
# n = 3
# path = find_all_paths(G, start_node, end_node, n)
# end_node = 8
# path = find_all_paths(G, start_node, end_node, n)
#
# if path:
#     print("找到路径：", path)
# else:
#     print("未找到路径")
#
# paths = nx.all_simple_paths(G, start_node, end_node, n)
# for p in paths:
#     print(p)


# import torch
# from attention_mechanisms.se_module import SELayer
# # 低一个点
# x = torch.randn(10, 5, 2, 256)
# attn = SELayer(256)
# y = attn(x)
# print(y.shape)
# x = torch.randn(10, 5, 3, 256)
# y = attn(x)
# print(y.shape)
# x = torch.randn(1, 1, 2, 256)
# y = attn(x)
# print(y.shape)

# input_tensor = torch.randn(1, 2, 256)  # 输入通道数为 3，尺寸为 32x32
# conv2d = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=16, padding=1)
# out = conv2d(input_tensor)
# print(out.shape)
# input_tensor = torch.randn(1, 3, 256)  # 输入通道数为 3，尺寸为 32x32
# out = conv2d(input_tensor)
# print(out.shape)

# from attention_mechanisms.cbam import CBAM
# 不太行
# x = torch.randn(10, 5, 2, 256)
# attn = CBAM(5)
# y = attn(x)
# print(y.shape)


# from attention_mechanisms.bam import BAM
# 不太行
# x = torch.randn(10, 5, 2, 256)
# attn = BAM(5)
# y = attn(x)
# print(y.shape)


# import torch
# from attention_mechanisms.srm import SRM
# 预测维度复杂
# x = torch.randn(10, 5, 2, 256)
# attn = SRM(5)
# y = attn(x)
# print(y.shape)
# x = torch.randn(10, 5, 3, 256)
# y = attn(x)
# print(y.shape)
# x = torch.randn(2, 5, 2, 256)
# y = attn(x)
# print(y.shape)


# import torch
# from attention_mechanisms.double_attention import DoubleAttention
# 完全不行
# x = torch.randn(10, 5, 2, 256)
# attn = DoubleAttention(5, 3, 256)
# y = attn(x)
# print(y.shape)
# x = torch.randn(10, 5, 2, 256)
# y = attn(x)
# print(y.shape)
# x = torch.randn(1, 5, 2, 256)
# y = attn(x)
# print(y.shape)


# import torch
# from attention_mechanisms.gate_channel_module import GCT
#
# attn = GCT(5)
# x = torch.randn(10, 5, 2, 256)
# y = attn(x)
# print(y.shape)
# x = torch.randn(10, 5, 3, 256)
# y = attn(x)
# print(y.shape)
# x = torch.randn(1, 5, 2, 256)
# y = attn(x)
# print(y.shape)

# import torch
#
# # 创建一个维度为 (a, b, c) 的张量
# tensor = torch.randn(2, 3, 4)
# print(tensor)
#
# # 使用 repeat 函数扩展维度为 (a, 5, b, c)
# expanded_tensor = tensor.unsqueeze(1).repeat(1, 5, 1, 1)
#
# print(expanded_tensor)  # 输出扩展后张量的大小

