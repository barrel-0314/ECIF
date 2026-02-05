import torch
import scipy
import pandas as pd
import pdb
import argparse
import pickle
import os
import numpy as np
import random
import tqdm
import time

from sympy.polys.polyconfig import query

from TAR import *
from my import *
from utils import *
from graph import *


class QuerySample(object):
    def __init__(self, model, graph, threshold=0.95):
        self.model = model              # 模型
        self.graph = graph              # 图
        self.threshold = threshold      # 正确度阈值     init
        self.query_type = None          # 查询类型
        self.pos = None                 # 查询
        self.ratio = None               # topk比例
        self.ranks = None               # 排名集合       preQuery
        self.epoch = 500                # 单轮采样次数
        self.ranks_set = None           # 排名前列的集合ranks[:ratio*size]
        self.sims = {}                  # 相似度字典
        self.visited = set()            # 已访问节点集合
        self.uni_info = {}              # 全局统计信息
        self.prop = None                # 查询属性
        self.step = 1                   # 遍历节点数
        self.stack = None               # 路径/充当栈
        self.k = None                   # 采样空间大小    approximate_compute
        self.targets = None
        self.answers = None

    def pre_query_by_model(self, query_type, mix, pos, ratio=0.02):
        self.query_type = query_type
        self.pos = pos[0].tolist()
        self.ratio = ratio
        logits = self.model.predict(mix, query_type=query_type, answer_type='e')
        self.ranks = torch.argsort(logits.squeeze(dim=0), descending=True).tolist()

    def sample(self):
        self.dfs()

    def calculate(self, op="count"):
        targets = []
        answers = []
        self.uni_info = {"sum": 0, "count": 0, "X2": 0}
        for k, v in self.sims.items():
            if v > self.threshold:
                if op == "count":
                    targets.append((k, round(v, 3)))
                    answers.append(k)
                    self.uni_info["sum"] += 1
                    self.uni_info["count"] += 1
                    self.uni_info["X2"] += 1
                elif self.prop is None:
                    targets.append((k, round(v, 3)))
                    answers.append(k)
                elif self.graph[k][self.prop] is not None:
                    val = self.graph[k][self.prop]
                    targets.append((k, round(v, 3)))
                    answers.append(k)
                    self.uni_info["sum"] += val
                    self.uni_info["count"] += 1
                    self.uni_info["X2"] += val * val
        self.targets = targets
        self.answers = answers
        return {"targets": targets, "answers": answers}

    def estimate(self, op="count", e=0.05, confidence=0.95):
        n_digits = 3
        uni_count = int(self.uni_info["count"])
        uni_sum = float(self.uni_info["sum"])
        uni_X2 = float(self.uni_info["X2"])

        traversed = sum(1 for value in self.sims.values() if value != 0)
        if traversed == 0:
            print("当前无正确节点，继续采样")
            return False, None, None
            # traversed = 1
        # TODO:此处公式有问题，分母应为ranks[0:k]和子图所节点数量交集的大小
        v_count = (uni_count / traversed) * self.k  # (正确节点数量cnt / 集合内遍历节点总数(sims!=0)) * 子图节点总数k
        var_count = (uni_count / traversed * self.k ** 2) - (v_count ** 2)
        std_count = math.sqrt(var_count)
        v_easy = sum(1 for value in self.sims.values() if value > 0.99)

        y = (1 + confidence) / 2
        x = scipy.stats.norm(0, 1).ppf(y)  # X~N(μ,σ^2)
        tmp = x / math.sqrt(traversed)
        epsilon_count = tmp * std_count
        epsilon_count = round(epsilon_count, n_digits)

        if op == "count":
            if epsilon_count <= v_count * e:
                # print('满足精度要求')
                return True, round(v_count, n_digits), epsilon_count, v_easy
            else:
                # print('不满足精度要求')
                return False, round(v_count, n_digits), epsilon_count, v_easy

    def approximate_compute(self, true_res, op="count", epoch=500, prop=None):
        """
        近似查询主函数
        :param true_res: 真实结果，count为数量，sum，avg为具体值
        :param op: 查询类型，count，sum，avg
        :param epoch: 可选参数，每轮采样步数
        :param prop: 可选参数，查询属性，count可不填，sum和avg得有
        :return:
        """
        self.epoch = epoch
        self.ranks_set = set(self.ranks[:int(len(self.model.e_dict) * self.ratio)])
        self.sims = {key: 0 for key in self.ranks_set}
        self.uni_info = {"sum": 0, "count": 0, "X2": 0}
        self.prop = prop
        self.step = 1

        n_digits = 3
        sample_round = 0
        node, _ = get_source(self.query_type, self.pos)
        subgraph_nodes = get_subgraph_nodes(self.graph, node, self.query_type, True)
        self.visited = set()
        self.stack = [[node]]
        self.k = len(self.ranks_set.intersection(subgraph_nodes))

        while True:
            sample_round += 1
            if not self.stack:
                print("子图已遍历完，无法继续采样")
                break
            self.sample()
            self.calculate()
            est_info = self.estimate(op)

            if est_info[0]:
                break

            final_res = est_info[1]
            if not final_res:
                continue
            final_epsilon = est_info[2]
            err_rate = abs(true_res - final_res) / true_res
            v_easy = est_info[3]
            err_rate_easy = abs(true_res - v_easy) / true_res

            print(f'{sample_round} 轮预测结果: {op} = {round(final_res, n_digits)}, 精确结果 =', true_res, ', 误差率',
                  '{:.2%}'.format(err_rate), f' ### 对比：纯遍历方法结果: {op} = {v_easy}, 误差率', '{:.2%}'.format(err_rate_easy))
            print(f'置信区间: [{round(final_res - final_epsilon, n_digits)}, {round(final_res + final_epsilon, n_digits)}]',
                  f'误差边界: {final_epsilon}')
            if true_res > final_res + final_epsilon or true_res < final_res - final_epsilon:
                print('未落入置信区间')
            else:
                print('成功落入置信区间')

    # 非递归dfs，以path替代栈
    def dfs(self):
        while self.stack:
            # print("visited:", self.visited, "\tstack:", self.stack)
            if len(self.stack) >= 2 and len(self.stack[-1]) <= 3 and len(self.stack[-1]) > len(self.stack[-2]):
                # self.visited.remove(self.stack[-1][-1])
                pass
            path = self.stack.pop()
            self.step += 1
            node = path[-1]

            # 达到指定数量，退出本次采样
            if self.step % self.epoch == 0:
                print(str(self.step) + " ", path)
                return

            # 当层处理逻辑，路径正确性判断
            if self.query_type == '1p' and len(path) == 2:
                source, r = self.pos[5], self.pos[6]
                target = path[1]
                if target in self.ranks_set:
                    sim = cos_sim(model, r, graph[source][target]['r_dict'])
                    self.sims[target] = max(sim, self.sims[target])

            if self.query_type == '2p' and len(path) == 3:
                source, r1, r2 = self.pos[4], self.pos[5], self.pos[6]
                mid, target = path[1], path[2]
                if target in self.ranks_set:
                    pr1, pr2 = graph[source][mid]['r_dict'], graph[mid][target]['r_dict']
                    sim1, sim2 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2)
                    sim = geo_mean([sim1, sim2])
                    self.sims[target] = max(round(sim, 3), self.sims[target])

            if self.query_type == '3p' and len(path) == 4:
                source, r1, r2, r3 = self.pos[3], self.pos[4], self.pos[5], self.pos[6]
                mid1, mid2, target = path[1], path[2], path[3]
                if target in self.ranks_set:
                    pr1, pr2, pr3 = graph[source][mid1]['r_dict'], graph[mid1][mid2]['r_dict'], \
                                    graph[mid2][target]['r_dict']
                    sim1, sim2, sim3 = cos_sim(model, r1, pr1), cos_sim(model, r2, pr2), cos_sim(model, r3, pr3)
                    sim = geo_mean([sim1, sim2, sim3])
                    self.sims[target] = max(round(sim, 3), self.sims[target])

            # 走到这里说明已经判断完所有查询类型，此时长度达到限度便可退出
            if len(path) == 4:
                continue

            self.visited.add(node)  # 防止成环
            for neighbor in reversed(list(graph.neighbors(node))):
                if neighbor not in self.visited:
                    self.stack.append(path + [neighbor])


if __name__ == '__main__':
    # 基本配置
    # save_root = '../models/13/'
    # model_path = save_root + str(800)
    # dataset = "DBpedia"
    # cfg = parse_args(["--dataset", dataset, "--emb_dim", "768"])
    save_root = '../models/DBpedia/initial/'
    model_path = save_root + str(900)
    dataset = "DBpedia"
    cfg = parse_args(["--dataset", dataset])
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    e_dict, c_dict, r_dict, ot, is_data_train = get_mapper(cfg.root + cfg.dataset + '/')

    # 加载模型
    model = TAR(cfg.emb_dim, e_dict, c_dict, r_dict)
    model.load_state_dict(torch.load(model_path))

    # 构建图
    edges = build_edges_incomplete(e_dict, r_dict, dataset)
    nodes = build_nodes(e_dict, dataset)
    graph = build_graph(nodes, edges)

    # 加载数据集
    train_e_1p, train_c_1p, train_filter_e_1p, train_filter_c_1p, test_e_1p, test_c_1p, test_filter_e_1p, test_filter_c_1p, valid_dataloader_1p_e, valid_dataloader_1p_c, test_dataloader_1p_e, test_dataloader_1p_c = load_train_and_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='1p')
    train_e_2p, train_c_2p, train_filter_e_2p, train_filter_c_2p, test_e_2p, test_c_2p, test_filter_e_2p, test_filter_c_2p, valid_dataloader_2p_e, valid_dataloader_2p_c, test_dataloader_2p_e, test_dataloader_2p_c = load_train_and_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2p')
    train_e_3p, train_c_3p, train_filter_e_3p, train_filter_c_3p, test_e_3p, test_c_3p, test_filter_e_3p, test_filter_c_3p, valid_dataloader_3p_e, valid_dataloader_3p_c, test_dataloader_3p_e, test_dataloader_3p_c = load_train_and_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3p')
    train_e_2i, train_c_2i, train_filter_e_2i, train_filter_c_2i, test_e_2i, test_c_2i, test_filter_e_2i, test_filter_c_2i, valid_dataloader_2i_e, valid_dataloader_2i_c, test_dataloader_2i_e, test_dataloader_2i_c = load_train_and_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2i')
    train_e_3i, train_c_3i, train_filter_e_3i, train_filter_c_3i, test_e_3i, test_c_3i, test_filter_e_3i, test_filter_c_3i, valid_dataloader_3i_e, valid_dataloader_3i_c, test_dataloader_3i_e, test_dataloader_3i_c = load_train_and_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3i')
    test_e_pi, test_c_pi, test_filter_e_pi, test_filter_c_pi, valid_dataloader_pi_e, valid_dataloader_pi_c, test_dataloader_pi_e, test_dataloader_pi_c = load_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='pi')
    test_e_ip, test_c_ip, test_filter_e_ip, test_filter_c_ip, valid_dataloader_ip_e, valid_dataloader_ip_c, test_dataloader_ip_e, test_dataloader_ip_c = load_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='ip')
    test_e_2u, test_c_2u, test_filter_e_2u, test_filter_c_2u, valid_dataloader_2u_e, valid_dataloader_2u_c, test_dataloader_2u_e, test_dataloader_2u_c = load_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2u')
    test_e_up, test_c_up, test_filter_e_up, test_filter_c_up, valid_dataloader_up_e, valid_dataloader_up_c, test_dataloader_up_e, test_dataloader_up_c = load_test_data(
        cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='up')

    # dataloader
    train_filters_e = {'1p': train_filter_e_1p, '2p': train_filter_e_2p, '3p': train_filter_e_3p}
    train_filters_c = {'1p': train_filter_c_1p, '2p': train_filter_c_2p, '3p': train_filter_c_3p}
    train_dataset = ValidDataset(train_e_1p, len(e_dict))
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=1, shuffle=False,
                                                   drop_last=False)

    # test
    query_type = '2p'
    loader, filters_e = dataloader(cfg, e_dict, c_dict, r_dict, query_type)
    gt = get_dict_gt(e_dict, r_dict, query_type)

    # 开启查询
    query = QuerySample(model, graph, 0.95)
    for index, (pos, mix) in enumerate(loader):
        if index > 100:
            break
        query.pre_query_by_model(query_type, mix, pos)
        # er = (pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
        # er = (pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())
        er = get_er(query_type, pos)
        query.approximate_compute(len(gt[er]))

    # dfs_iterative_limit_path_length(3087, graph)