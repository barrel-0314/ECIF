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


class JaccardCompare(object):
    def __init__(self, model, graph, cfg, e_dict, c_dict, r_dict, threshold=0.95):
        self.model = model  # 模型
        self.graph = graph  # 图
        self.cfg = cfg  # 命令行配置
        self.e_dict = e_dict  # 实体
        self.c_dict = c_dict  # 关系
        self.r_dict = r_dict  # 本体
        self.threshold = threshold  # 正确度阈值     init
        self.query_type = None  # 查询类型
        self.loader = None  # 数据加载器
        self.filters_e = None  # 答案          load_data
        self.pos = None  # 查询
        self.ratio = None  # topk比例
        self.ranks = None  # 排名集合       pre_query
        self.epoch = 500  # 单轮采样次数
        self.ranks_set = None  # 排名前列的集合ranks[:ratio*size]
        self.sims = {}  # 相似度字典
        self.visited = set()  # 已访问节点集合
        self.uni_info = {}  # 全局统计信息
        self.prop = None  # 查询属性
        self.step = 1  # 遍历节点数
        self.stack = None  # 路径/充当栈
        self.k = None  # 采样空间大小    approximate_compute
        self.targets = None
        self.answers = None

    def load_data(self, query_type):
        self.query_type = query_type
        self.loader, self.filters_e = dataloader(self.cfg, self.e_dict, self.c_dict, self.r_dict, query_type)

    def pre_query_by_model(self, query_type, mix, pos, ratio=0.01):
        self.query_type = query_type
        self.pos = pos[0].tolist()
        self.ratio = ratio
        self.k = int(len(self.model.e_dict) * ratio)
        logits = self.model.predict(mix, query_type=query_type, answer_type='e')
        self.ranks = torch.argsort(logits.squeeze(dim=0), descending=True)

    def cal_jaccard(self):
        start_time = time.time()
        jaccards, precision, recall = [], [], []
        for index, (pos, mix) in enumerate(self.loader):
            er = get_er(query_type, pos)
            filter_e = self.filters_e[er]
            self.pre_query_by_model(query_type, mix, pos, 0.02)
            targets, answers = validV2(self.ranks.tolist(), self.k, self.query_type, pos[0].tolist(), self.graph,
                                       self.model, self.threshold)
            print(er, ":", targets)
            print(er, ":", gt[er])
            rank = (self.ranks == (pos[0, -1])).nonzero().item() + 1
            ranks_better = self.ranks[:rank - 1]
            for t in filter_e:
                if (ranks_better == t).sum() == 1:
                    rank -= 1
            jaccards.append(jaccard_similarity(answers, gt[er]))
        end_time = time.time()
        run_time = end_time - start_time
        print(run_time)
        print("jaccard系数: ", np.mean(jaccards))


if __name__ == '__main__':
    # 基本配置
    # save_root = '../models/DBpedia/13/'
    # model_path = save_root + str(800)
    # dataset = "DBpedia"
    # cfg = parse_args(["--dataset", dataset, "--emb_dim", "768"])
    # initial
    save_root = '../models/DBpedia/initial/'
    model_path = save_root + str(900)
    # 12
    # save_root = '../models/DBpedia/12/'
    # model_path = save_root + str(1000)
    # 14
    # save_root = '../models/DBpedia/14/'
    # model_path = save_root + str(700)
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
    # train_e_1p, train_c_1p, train_filter_e_1p, train_filter_c_1p, test_e_1p, test_c_1p, test_filter_e_1p, test_filter_c_1p, valid_dataloader_1p_e, valid_dataloader_1p_c, test_dataloader_1p_e, test_dataloader_1p_c = load_train_and_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='1p')
    # train_e_2p, train_c_2p, train_filter_e_2p, train_filter_c_2p, test_e_2p, test_c_2p, test_filter_e_2p, test_filter_c_2p, valid_dataloader_2p_e, valid_dataloader_2p_c, test_dataloader_2p_e, test_dataloader_2p_c = load_train_and_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2p')
    # train_e_3p, train_c_3p, train_filter_e_3p, train_filter_c_3p, test_e_3p, test_c_3p, test_filter_e_3p, test_filter_c_3p, valid_dataloader_3p_e, valid_dataloader_3p_c, test_dataloader_3p_e, test_dataloader_3p_c = load_train_and_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3p')
    # train_e_2i, train_c_2i, train_filter_e_2i, train_filter_c_2i, test_e_2i, test_c_2i, test_filter_e_2i, test_filter_c_2i, valid_dataloader_2i_e, valid_dataloader_2i_c, test_dataloader_2i_e, test_dataloader_2i_c = load_train_and_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2i')
    # train_e_3i, train_c_3i, train_filter_e_3i, train_filter_c_3i, test_e_3i, test_c_3i, test_filter_e_3i, test_filter_c_3i, valid_dataloader_3i_e, valid_dataloader_3i_c, test_dataloader_3i_e, test_dataloader_3i_c = load_train_and_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3i')
    # test_e_pi, test_c_pi, test_filter_e_pi, test_filter_c_pi, valid_dataloader_pi_e, valid_dataloader_pi_c, test_dataloader_pi_e, test_dataloader_pi_c = load_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='pi')
    # test_e_ip, test_c_ip, test_filter_e_ip, test_filter_c_ip, valid_dataloader_ip_e, valid_dataloader_ip_c, test_dataloader_ip_e, test_dataloader_ip_c = load_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='ip')
    # test_e_2u, test_c_2u, test_filter_e_2u, test_filter_c_2u, valid_dataloader_2u_e, valid_dataloader_2u_c, test_dataloader_2u_e, test_dataloader_2u_c = load_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2u')
    # test_e_up, test_c_up, test_filter_e_up, test_filter_c_up, valid_dataloader_up_e, valid_dataloader_up_c, test_dataloader_up_e, test_dataloader_up_c = load_test_data(
    #     cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='up')

    # dataloader
    # train_filters_e = {'1p': train_filter_e_1p, '2p': train_filter_e_2p, '3p': train_filter_e_3p}
    # train_filters_c = {'1p': train_filter_c_1p, '2p': train_filter_c_2p, '3p': train_filter_c_3p}
    # train_dataset = ValidDataset(train_e_1p, len(e_dict))
    # train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=1, shuffle=False,
    #                                                drop_last=False)

    # test
    query_type = "2u"
    loader = dataloader(cfg, e_dict, c_dict, r_dict, query_type)
    gt = get_dict_gt(e_dict, r_dict, query_type)

    # 开启查询
    jaccard = JaccardCompare(model, graph, cfg, e_dict, c_dict, r_dict, 0.999)
    jaccard.load_data(query_type)
    jaccard.cal_jaccard()
