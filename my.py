import torch
import pandas as pd
import pdb
import argparse
import pickle
import os
import numpy as np
import random
from tqdm import tqdm
import sys

from transformers import BertTokenizer, BertModel
from utils import *

query_types = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2u', 'up']


# TODO:创建双向字典，从key到value不再遍历
# 从字典返回value对应的key
def get_e_by_dict(e_dict, num):
    for e, i in e_dict.items():
        if i == num:
            return e
    return None


# 从文件中根据查询类型返回key值，例如2p，返回(e,r1,r2)的形式
def get_key_by_type(e_dict, r_dict, query_type, keys):
    if query_type == '1p':
        key = (e_dict[keys[0]], r_dict[keys[1]])
    elif query_type == '2p':
        key = (e_dict[keys[0]], r_dict[keys[1]], r_dict[keys[2]])
    elif query_type == '3p':
        key = (e_dict[keys[0]], r_dict[keys[1]], r_dict[keys[2]], r_dict[keys[3]])
    elif query_type == '2i':
        key = (e_dict[keys[0]], r_dict[keys[1]], e_dict[keys[2]], r_dict[keys[3]])
    elif query_type == '3i':
        key = (e_dict[keys[0]], r_dict[keys[1]], e_dict[keys[2]], r_dict[keys[3]], e_dict[keys[4]], r_dict[keys[5]])
    elif query_type == 'pi':
        key = (e_dict[keys[0]], r_dict[keys[1]], r_dict[keys[2]], e_dict[keys[3]], r_dict[keys[4]])
    elif query_type == 'ip':
        key = (e_dict[keys[0]], r_dict[keys[1]], e_dict[keys[2]], r_dict[keys[3]], r_dict[keys[4]])
    elif query_type == '2u':
        key = (e_dict[keys[0]], r_dict[keys[1]], e_dict[keys[2]], r_dict[keys[3]])
    elif query_type == 'up':
        key = (e_dict[keys[0]], r_dict[keys[1]], e_dict[keys[2]], r_dict[keys[3]], r_dict[keys[4]])
    else:
        raise ValueError
    return key


# 获取所有答案集合，包括图中存在的和不存在的
# 数据集大坑，1p的test/e是集合，其他全是字符串
def get_dict_gt(e_dict, r_dict, query_type):
    data = load_obj('../data/DBpedia/input/' + query_type + '.pkl')
    dict_gt = {}
    for keys in data["test"]["e"]:
        # 遵循数据集构建，去除低度节点
        if keys[0] not in e_dict:
            continue
        ret = []
        if query_type == '1p':
            for answer in data["test"]["e"][keys]:
                ret.append(e_dict[answer])
        else:
            ret.append(e_dict[data["test"]["e"][keys]])
        for answer in data["test_filter"]["e"][keys]:
            ret.append(e_dict[answer])
        key = get_key_by_type(e_dict, r_dict, query_type, keys)
        ret.sort()
        dict_gt[key] = ret
    return dict_gt


def my_evaluate_e(model, loader, filters_e, device, query_type):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    h50 = []
    # if cfg.verbose == 1:
    #     loader = tqdm.tqdm(loader)
    with torch.no_grad():
        for pos, mix in loader:
            mix = mix.to(device)
            if query_type == '1p':
                logits = model.predict(mix, query_type='1p', answer_type='e')
                filter_e = filters_e[(pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2p':
                logits = model.predict(mix, query_type='2p', answer_type='e')
                filter_e = filters_e[(pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3p':
                logits = model.predict(mix, query_type='3p', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2i':
                logits = model.predict(mix, query_type='2i', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3i':
                logits = model.predict(mix, query_type='3i', answer_type='e')
                filter_e = filters_e[(
                    pos[0, 1].item(), pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(),
                    pos[0, 6].item())]
            elif query_type == 'pi':
                logits = model.predict(mix, query_type='pi', answer_type='e')
                filter_e = filters_e[
                    (pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'ip':
                logits = model.predict(mix, query_type='ip', answer_type='e')
                filter_e = filters_e[
                    (pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2u':
                logits = model.predict(mix, query_type='2u', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'up':
                logits = model.predict(mix, query_type='up', answer_type='e')
                filter_e = filters_e[
                    (pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            else:
                raise ValueError
            ranks = torch.argsort(logits.squeeze(dim=0), descending=True)
            rank = (ranks == (pos[0, -1])).nonzero().item() + 1
            ranks_better = ranks[:rank - 1]
            for t in filter_e:
                if (ranks_better == t).sum() == 1:
                    rank -= 1
            r.append(rank)
            rr.append(1 / rank)
            if rank == 1:
                h1.append(1)
            else:
                h1.append(0)
            if rank <= 3:
                h3.append(1)
            else:
                h3.append(0)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
            if rank <= 50:
                h50.append(1)
            else:
                h50.append(0)
    r = int(sum(r) / len(r))
    rr = round(sum(rr) / len(rr), 3)
    h1 = round(sum(h1) / len(h1), 3)
    h3 = round(sum(h3) / len(h3), 3)
    h10 = round(sum(h10) / len(h10), 3)
    h50 = round(sum(h50) / len(h50), 3)
    print(f'#Entity#{query_type}# MRR: {rr}, H1: {h1}, H3: {h3}, H10: {h10}, H50: {h50}')
    return r, rr, h1, h3, h10, h50


# 针对不同查询类型，验证所有答案的rank，从而确定一个合适的k，作为实际聚集查询时的候选答案集
def my_evaluate(model, test_gt, loader, device, query_type):
    res = []
    with torch.no_grad():
        for pos, mix in loader:
            mix = mix.to(device)
            if query_type == '1p':
                logits = model.predict(mix, query_type='1p', answer_type='e')
                gt = test_gt[(pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2p':
                logits = model.predict(mix, query_type='2p', answer_type='e')
                gt = test_gt[(pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3p':
                logits = model.predict(mix, query_type='3p', answer_type='e')
                gt = test_gt[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2i':
                logits = model.predict(mix, query_type='2i', answer_type='e')
            elif query_type == '3i':
                logits = model.predict(mix, query_type='3i', answer_type='e')
            elif query_type == 'pi':
                logits = model.predict(mix, query_type='pi', answer_type='e')
            elif query_type == 'ip':
                logits = model.predict(mix, query_type='ip', answer_type='e')
            elif query_type == '2u':
                logits = model.predict(mix, query_type='2u', answer_type='e')
            elif query_type == 'up':
                logits = model.predict(mix, query_type='up', answer_type='e')
            else:
                raise ValueError
            ranks = torch.argsort(logits.squeeze(dim=0), descending=True).tolist()
            indices = [ranks.index(x) for x in gt]
            res.append(indices)
        return res


# 根据诡异的数据格式计算起点用于p类查询的dfs
def get_source(query_type, pos):
    if query_type == "1p":
        return pos[5], None
    elif query_type == "2p":
        return pos[4], None
    elif query_type == "3p":
        return pos[3], None
    elif query_type == "2i":
        return None, None
    elif query_type == "3i":
        return None, None
    elif query_type == "pi":
        return pos[2], pos[5]
    elif query_type == "ip":
        return pos[2], pos[4]
    elif query_type == "up":
        return pos[2], pos[4]
    elif query_type == "2u":
        return None, None
    elif query_type == "up":
        return None, None


def get_er(query_type, pos):
    if query_type == '1p':
        return (pos[0, 5].item(), pos[0, 6].item())
    elif query_type == '2p':
        return (pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
    elif query_type == '3p':
        return (pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
    elif query_type == '2i':
        return (pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
    elif query_type == '3i':
        return (pos[0, 1].item(), pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
    elif query_type == 'pi':
        return (pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
    elif query_type == 'ip':
        return (pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
    elif query_type == '2u':
        return (pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
    elif query_type == 'up':
        return (pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())


def valid_optimize(ranks, k, query_type, pos, graph, model, threshold):
    # sims = []   # 相似度
    # targets = []    # 结果与相似度
    # answers = []    # 结果
    start = get_source(query_type, pos)
    ranks_set = set(ranks[:k])
    sims = {key: 0 for key in ranks_set}

    # paths = []
    visited = set()

    def dfs(node, path):
        # print(path)
        if node not in graph:
            return
        if query_type == '1p' and len(path) == 2:
            source, r = pos[5], pos[6]
            target = path[1]
            if target in ranks_set:
                sim = cos_sim(model, r, graph[source][target]['r_dict'])
                sims[target] = max(sim, sims[target])

        if query_type == '2p' and len(path) == 3:
            source, r1, r2 = pos[4], pos[5], pos[6]
            mid, target = path[1], path[2]
            if target in ranks_set:
                pr1, pr2 = graph[source][mid]['r_dict'], graph[mid][target]['r_dict']
                sim1, sim2 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2)
                sim = geo_mean([sim1, sim2])
                sims[target] = max(round(sim, 3), sims[target])

        if query_type == '3p' and len(path) == 4:
            source, r1, r2, r3 = pos[3], pos[4], pos[5], pos[6]
            mid1, mid2, target = path[1], path[2], path[3]
            if target in ranks_set:
                pr1, pr2, pr3 = graph[source][mid1]['r_dict'], graph[mid1][mid2]['r_dict'], \
                                graph[mid2][target]['r_dict']
                sim1, sim2, sim3 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2), cos_sim(model, r3, pr3)
                sim = geo_mean([sim1, sim2, sim3])
                sims[target] = max(round(sim, 3), sims[target])

        # 走到这里说明已经判断完所有查询类型，此时长度达到限度便可退出
        if len(path) == 4:
            return

        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
        visited.remove(node)

    # 执行dfs，更新获得相似度字典
    if query_type.endswith("p"):
        dfs(start, [start])
    elif query_type == "2i":
        e1, r1, e2, r2 = pos[3], pos[4], pos[5], pos[6]
        if graph.has_node(e1) and graph.has_node(e2):
            sims1 = {neighbor: cos_sim(model, r1, graph[e1][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e1) if neighbor in ranks_set}
            sims2 = {neighbor: cos_sim(model, r2, graph[e2][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e2) if neighbor in ranks_set}
            # sims1 = {neighbor: cos_sim(model, r1, graph[e1][neighbor]['r_dict'])
            #          for neighbor in ranks_set}
            # sims2 = {neighbor: cos_sim(model, r2, graph[e2][neighbor]['r_dict'])
            #          for neighbor in ranks_set}
            intersection = sims1.keys() & sims2.keys()
            sims = {i: geo_mean([sims1[i], sims2[i]]) for i in intersection}
    elif query_type == "3i":
        e1, r1, e2, r2, e3, r3 = pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]
        if graph.has_node(e1) and graph.has_node(e2) and graph.has_node(e3):
            sims1 = {neighbor: cos_sim(model, r1, graph[e1][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e1) if neighbor in ranks_set}
            sims2 = {neighbor: cos_sim(model, r2, graph[e2][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e2) if neighbor in ranks_set}
            sims3 = {neighbor: cos_sim(model, r3, graph[e3][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e3) if neighbor in ranks_set}
            intersection = sims1.keys() & sims2.keys() & sims3.keys()
            sims = {i: geo_mean([sims1[i], sims2[i], sims3[i]]) for i in intersection}

    targets = []
    answers = []
    for k, v in sims.items():
        if v > threshold:
            targets.append((k, round(v, 3)))
            answers.append(k)
    return targets, answers


def validV2(ranks, k, query_type, pos, graph, model, threshold):
    start1, start2 = get_source(query_type, pos)    # 1p~3p只有1个锚节点，pi、ip、up有两个
    ranks_set = set(ranks[:k])
    sims1, sims2 = {key: 0 for key in ranks_set}, {key: 0 for key in ranks_set}

    # paths = []
    visited = set()

    def dfs(node, path):
        if node not in graph:
            return
        target = path[-1]
        # print(path)
        if target in ranks_set:
            if query_type == "1p":
                if len(path) == 2:
                    r = pos[6]
                    sim = cos_sim(model, r, graph[start1][target]['r_dict'])
                    sims1[target] = max(sim, sims1[target])
                # elif len(path) == 3:
                #     mid = path[-2]
                #     sim1 = cos_sim(model, r, graph[start][mid]['r_dict'])
                #     sim2 = cos_sim(model, r, graph[mid][target]['r_dict'])
                #     sims[target] = max(geo_mean([sim1, sim2]), sims[target])
                # elif len(path) == 4:
                #     mid1 = path[-3]
                #     mid2 = path[-2]
                #     sim1 = cos_sim(model, r, graph[start][mid1]['r_dict'])
                #     sim2 = cos_sim(model, r, graph[mid1][mid2]['r_dict'])
                #     sim3 = cos_sim(model, r, graph[mid2][target]['r_dict'])
                #     sims[target] = max(geo_mean([sim1, sim2, sim3]), sims[target])
            elif query_type == "2p":
                if len(path) == 3:
                    r1, r2 = pos[5], pos[6]
                    mid = path[-2]
                    pr1, pr2 = graph[start1][mid]['r_dict'], graph[mid][target]['r_dict']
                    sim1, sim2 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2)
                    sim = geo_mean([sim1, sim2])
                    sims1[target] = max(round(sim, 3), sims1[target])
            elif query_type == '3p':
                if len(path) == 4:
                    r1, r2, r3 = pos[4], pos[5], pos[6]
                    mid1, mid2 = path[-3], path[-2]
                    pr1, pr2, pr3 = graph[start1][mid1]['r_dict'], graph[mid1][mid2]['r_dict'], \
                        graph[mid2][target]['r_dict']
                    sim1, sim2, sim3 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2), cos_sim(model, r3, pr3)
                    sim = geo_mean([sim1, sim2, sim3])
                    sims1[target] = max(round(sim, 3), sims1[target])
            elif query_type == "pi":
                if start1 == path[0] and len(path) == 3:
                    mid = path[-2]
                    r1, r2 = pos[3], pos[4]
                    pr1, pr2 = graph[start1][mid]['r_dict'], graph[mid][target]['r_dict']
                    sim1, sim2 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2)
                    sim = geo_mean([sim1, sim2])
                    sims1[target] = max(round(sim, 3), sims1[target])
                elif start2 == path[0] and len(path) == 2:
                    r = pos[6]
                    if len(path) == 2:
                        sim = cos_sim(model, r, graph[start2][target]['r_dict'])
                        sims2[target] = max(sim, sims2[target])
            elif query_type == "ip" or query_type == "up":
                if start1 == path[0] and len(path) == 3:
                    mid = path[-2]
                    r1, r2 = pos[3], pos[6]
                    pr1, pr2 = graph[start1][mid]['r_dict'], graph[mid][target]['r_dict']
                    sim1, sim2 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2)
                    sim = geo_mean([sim1, sim2])
                    sims1[target] = max(round(sim, 3), sims1[target])
                if start2 == path[0] and len(path) == 3:
                    mid = path[-2]
                    r1, r2 = pos[5], pos[6]
                    pr1, pr2 = graph[start2][mid]['r_dict'], graph[mid][target]['r_dict']
                    sim1, sim2 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2)
                    sim = geo_mean([sim1, sim2])
                    sims2[target] = max(round(sim, 3), sims2[target])

        if len(path) == 4:
            return
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
        visited.remove(node)

    if query_type in query_types[0:3]:
        dfs(start1, [start1])
    elif query_type == "2i":
        e1, r1, e2, r2 = pos[3], pos[4], pos[5], pos[6]
        if graph.has_node(e1) and graph.has_node(e2):
            sims1 = {neighbor: cos_sim(model, r1, graph[e1][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e1) if neighbor in ranks_set}
            sims2 = {neighbor: cos_sim(model, r2, graph[e2][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e2) if neighbor in ranks_set}
            intersection = sims1.keys() & sims2.keys()
            sims1 = {i: geo_mean([sims1[i], sims2[i]]) for i in intersection}
    elif query_type == "3i":
        e1, r1, e2, r2, e3, r3 = pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]
        if graph.has_node(e1) and graph.has_node(e2) and graph.has_node(e3):
            sims1 = {neighbor: cos_sim(model, r1, graph[e1][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e1) if neighbor in ranks_set}
            sims2 = {neighbor: cos_sim(model, r2, graph[e2][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e2) if neighbor in ranks_set}
            sims3 = {neighbor: cos_sim(model, r3, graph[e3][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e3) if neighbor in ranks_set}
            intersection = sims1.keys() & sims2.keys() & sims3.keys()
            sims1 = {i: geo_mean([sims1[i], sims2[i], sims3[i]]) for i in intersection}
    elif query_type == "pi":
        dfs(start1, [start1])
        dfs(start2, [start2])
        for k in sims1.keys():
            sims1[k] = geo_mean([sims1[k], sims2[k]])
    elif query_type == "ip":
        dfs(start1, [start1])
        dfs(start2, [start2])
        for k in sims1.keys():
            sims1[k] = geo_mean([sims1[k], sims2[k]])
    elif query_type == "2u":
        e1, r1, e2, r2 = pos[3], pos[4], pos[5], pos[6]
        if graph.has_node(e1) and graph.has_node(e2):
            sims1 = {neighbor: cos_sim(model, r1, graph[e1][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e1) if neighbor in ranks_set}
            sims2 = {neighbor: cos_sim(model, r2, graph[e2][neighbor]['r_dict'])
                     for neighbor in graph.neighbors(e2) if neighbor in ranks_set}
            union = sims1.keys() | sims2.keys()
            sims1 = {i: max(sims1.get(i, 0), sims1.get(i, 0)) for i in union}
    elif query_type == "up":
        dfs(start1, [start1])
        dfs(start2, [start2])
        for k in sims1.keys():
            sims1[k] = max(sims1[k], sims2[k])

    targets = []
    answers = []
    for k, v in sims1.items():
        if v > threshold:
            targets.append((k, round(v, 3)))
            answers.append(k)
    return targets, answers

# 从dataframe中构建类型层次多叉树，返回多叉树字典tree和根节点集合roots
def build_tree_from_dataframe(df):
    data = df[['t', 'h']].values.tolist()
    tree = {}
    child_nodes = set()

    for parent, child in data:
        child_nodes.add(child)  # 将所有子节点添加到集合中
        if parent not in tree:
            tree[parent] = []
        tree[parent].append(child)

    # 查找根节点
    roots = [node for node in tree.keys() if node not in child_nodes]

    return tree, roots


def find_paths(tree, node, path, paths):
    if node not in tree:
        paths.append(path + [node])
        return
    for child in tree[node]:
        find_paths(tree, child, path + [node], paths)


def find_paths_from_roots(tree, roots):
    paths = []
    for root in roots:
        find_paths(tree, root, [], paths)
    return paths


def embed_path(path):
    # 加载预训练的BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('../models/bert/', local_files_only=True)
    model = BertModel.from_pretrained('../models/bert/', local_files_only=True)
    # 使用分词器将节点编码成标记
    # tokens = [tokenizer.cls_token_id] + path + [tokenizer.sep_token_id]
    tokens = path
    input_ids = torch.tensor([tokens])
    # 使用BERT模型对路径进行编码
    with torch.no_grad():
        outputs = model(input_ids)
        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state
    # 返回路径上每个节点的嵌入向量
    return last_hidden_state.squeeze(0)


#
def concept_tree_init(c_dict, c_embedding):
    data = pd.read_csv("../data/DBpedia/mid/tree_init_embeddings.csv", header=None)
    for i in range(len(c_dict)):
        tensor = torch.tensor(eval(data[1][i]))   # 此处1表示列
        if len(tensor) > 0:
            c_embedding[i] = tensor
    # data = pd.read_csv('../data/DBpedia/mid/ot.csv')
    # data = data.drop(data.columns[0], axis=1)
    # tree, roots = build_tree_from_dataframe(data)
    # paths = find_paths_from_roots(tree, roots)
    # mapped_paths = [[c_dict[node] for node in path] for path in paths]
    # mapped_paths = torch.tensor(mapped_paths).to('cuda')
    # mapped_embeds = {key: [] for key in c_dict.values()}
    # bert嵌入，添加到字典（计算时间较长，改为处理后从文件读取）
    # for path in tqdm(mapped_paths):
    #     embeddings = embed_path(path)
    #     for i in range(len(embeddings)):
    #         node = path[i]
    #         mapped_embeds[node].append(embeddings[i])

    # 计算均值
    # for k, v in mapped_embeds.items():
    #     if (len(v) == 0):
    #         break
    #     avg = sum(v) / len(v)
    #     # mapped_embeds[k] = avg
    #     c_embedding[k] = avg


def dfs_iterative_limit_path_length(start_node, graph):
    stack = [(start_node, [start_node])]  # 使用元组 (node, path) 来表示节点和路径
    visited_traverse = set()
    visited_path = set()

    while stack:
        node, path = stack.pop()
        print(path)

        # 判断路径长度是否超过4，如果超过则不再处理该节点及其邻居节点
        if len(path) >= 4:
            continue

        if node not in visited_traverse:
            visited_traverse.add(node)
            # 处理当前节点
            # print(f"Visiting node: {node}, Path: {path}")

            # 将邻居节点按照相反的顺序推入栈中，以便先处理最深的节点
            for neighbor in list(reversed(list(graph.neighbors(node)))):
                if neighbor not in visited_path:
                    stack.append((neighbor, path + [neighbor]))

    print(visited_traverse)


def non_recursive_dfs(graph, start):
    # Stack to maintain the current path and visited nodes in this path
    stack = [(start, [start])]
    # Set to track visited nodes within the current path
    visited_in_path = set([start])
    # Set to track visited nodes across different paths
    visited = set()

    while stack:
        (vertex, path) = stack.pop()

        # Process node
        print("Current path:", path)

        # Check if path length exceeds 4
        if len(path) > 4:
            continue

        # Add current node to global visited set
        visited.add(vertex)

        # Explore neighbors
        for neighbor in graph[vertex]:
            if neighbor not in visited_in_path:
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))
                visited_in_path.add(neighbor)
            else:
                print(f"Node {neighbor} already visited in this path.")

    return visited
