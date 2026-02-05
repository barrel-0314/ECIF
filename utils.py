import pickle
import torch
import math
from torch.nn.functional import cosine_similarity
import networkx as nx
import numpy as np

def read_file(path, len):
    count = 0
    with open(path) as f:
        for line in f:
            if count < len:
                print(line)
                count += 1
            else:
                break


def cos_sim(model, r1, r2):
    w1 = model.r_embedding.weight[r1].unsqueeze(0)
    w2 = model.r_embedding.weight[r2].unsqueeze(0)
    return round(cosine_similarity(w1, w2).item(), 3)


def geo_mean(sims):
    # 防止负数开根出了复数报错
    return np.prod(sims) ** (1 / len(sims)) if np.prod(sims) > 0 else 0


def jaccard_similarity(list1, list2):
    # 将列表转换为集合
    set1 = set(list1)
    set2 = set(list2)
    # 计算交集大小
    intersection_size = len(set1.intersection(set2))
    # 计算并集大小
    union_size = len(set1.union(set2))
    # 计算杰卡德相似度
    similarity = intersection_size / union_size
    return similarity


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def tensor2val(val):
    if isinstance(val, torch.Tensor):
        return val.item()
    else:
        return val

def multiPathSim(query, path, threshold):
    i, j = 1, 1
    curSim = 1
    sims = []
    while j < len(path):
        edgeSim = cos_sim(path[i], query[j])
        curSim *= edgeSim
        sims.append(edgeSim)
        pss = geo_mean(sims)
        if pss > threshold:  # 进入下一个单跳查询
            i += 1
            j += 1
        else:
            if pow(curSim, 1/3) < threshold:  # 后续全匹配也无法满足精度，因此剪枝
                return
            else:
                j += 1





# 验证相似度匹配的可行性
# ranks 排名 k 取ranks前k名
def valid(ranks, k, query_type, pos, graph, model, threshold):
    sims = []   # 相似度
    targets = []    # 结果与相似度
    answers = []    # 结果
    if query_type == '1p':
        source, r = pos[5], pos[6]
        for rank in ranks[:k]:
            # 每个候选节点的最终相似度，由不同类型的路径相似度加权平均
            similarity = []
            # TODO: source和rank都可能不在图中，好像是ppc那边搞ontology改的，有空再解决
            if source not in graph or rank not in graph:
                # targets.append("节点不存在，待解决")
                continue
            paths = nx.all_simple_paths(graph, source, rank, 3)
            for path in paths:
                if len(path) == 2:
                    similarity.append(cos_sim(model, r, graph[source][path[1]]['r_dict']))
                    continue
                # if len(path) == 3:
                #     pr1, pr2 = graph[source][path[1]]['r_dict'], graph[path[1]][path[2]]['r_dict']
                #     sim1, sim2 = cos_sim(model, r, pr1), cos_sim(model, r, pr2)
                #     sim = (sim1 * sim2) ** (1/2) if (sim1 * sim2) > 0 else 0
                #     similarity.append(round(sim, 3))
                # if len(path) == 4:
                #     pr1, pr2, pr3 = graph[source][path[1]]['r_dict'], graph[path[1]][path[2]]['r_dict'], graph[path[2]][path[3]]['r_dict']
                #     sim1, sim2, sim3 = cos_sim(model, r, pr1), cos_sim(model, r, pr2), cos_sim(model, r, pr3)
                #     sim = (sim1 * sim2 * sim3) ** (1/3) if (sim1 * sim2 * sim3) > 0 else 0  # 负数开根出了复数报错
                #     similarity.append(round(sim, 3))
            max_sim = np.max(similarity) if len(similarity) > 0 else 0
            sims.append(max_sim)
            if max_sim > threshold:
                targets.append((rank, max_sim))
                answers.append(rank)

    if query_type == '2p':
        source, r1, r2 = pos[4], pos[5], pos[6]
        for rank in ranks[:k]:
            similarity = []
            if source not in graph or rank not in graph:
                # targets.append("节点不存在，待解决")
                continue
            paths = nx.all_simple_paths(graph, source, rank, 3)
            for path in paths:
                if len(path) == 3:
                    pr1, pr2 = graph[source][path[1]]['r_dict'], graph[path[1]][path[2]]['r_dict']
                    sim1, sim2 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2)
                    sim = (sim1 * sim2) ** (1 / 2) if (sim1 * sim2) > 0 else 0
                    similarity.append(round(sim, 3))
            max_sim = np.max(similarity) if len(similarity) > 0 else 0
            sims.append(max_sim)
            if max_sim > threshold:
                targets.append((rank, max_sim))
                answers.append(rank)

    if query_type == '3p':
        source, r1, r2, r3 = pos[3], pos[4], pos[5], pos[6]
        for rank in ranks[:k]:
            similarity = []
            if source not in graph or rank not in graph:
                # targets.append("节点不存在，待解决")
                continue
            paths = nx.all_simple_paths(graph, source, rank, 3)
            for path in paths:
                if len(path) == 4:
                    pr1, pr2, pr3 = graph[source][path[1]]['r_dict'], graph[path[1]][path[2]]['r_dict'], graph[path[2]][path[3]]['r_dict']
                    sim1, sim2, sim3 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2), cos_sim(model, r3, pr3)
                    sim = (sim1 * sim2 * sim3) ** (1/3) if (sim1 * sim2 * sim3) > 0 else 0  # 负数开根出了复数报错
                    similarity.append(round(sim, 3))
            max_sim = np.max(similarity) if len(similarity) > 0 else 0
            sims.append(max_sim)
            if max_sim > threshold:
                targets.append((rank, max_sim))
                answers.append(rank)
    return sims, targets, answers






