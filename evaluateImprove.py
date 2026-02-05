import torch
import pandas as pd
import pdb
import argparse
import pickle
import os
import numpy as np
import random
import tqdm
import time
from TAR import *
from my import *
from utils import *
from graph import *

save_root = '../models/5/'
model_path = save_root + str(900)
cfg = parse_args(["--dataset","DBpedia"])
seed_everything(cfg.seed)
device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
e_dict, c_dict, r_dict, ot, is_data_train = get_mapper(cfg.root + cfg.dataset + '/')

model = TAR(cfg.emb_dim, e_dict, c_dict, r_dict)
model.load_state_dict(torch.load(model_path))

edges = build_edges_incomplete(e_dict, r_dict)
nodes = build_nodes(e_dict)
graph = build_graph(nodes, edges)

train_ot = ppc(ot, e_dict, c_dict, r_dict, query_type='ot', answer_type=None, flag=None)
train_is = ppc(is_data_train, e_dict, c_dict, r_dict, query_type='is', answer_type=None, flag=None)

train_e_1p, train_c_1p, train_filter_e_1p, train_filter_c_1p, test_e_1p, test_c_1p, test_filter_e_1p, test_filter_c_1p, valid_dataloader_1p_e, valid_dataloader_1p_c, test_dataloader_1p_e, test_dataloader_1p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='1p')
train_e_2p, train_c_2p, train_filter_e_2p, train_filter_c_2p, test_e_2p, test_c_2p, test_filter_e_2p, test_filter_c_2p, valid_dataloader_2p_e, valid_dataloader_2p_c, test_dataloader_2p_e, test_dataloader_2p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2p')
train_e_3p, train_c_3p, train_filter_e_3p, train_filter_c_3p, test_e_3p, test_c_3p, test_filter_e_3p, test_filter_c_3p, valid_dataloader_3p_e, valid_dataloader_3p_c, test_dataloader_3p_e, test_dataloader_3p_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3p')
train_e_2i, train_c_2i, train_filter_e_2i, train_filter_c_2i, test_e_2i, test_c_2i, test_filter_e_2i, test_filter_c_2i, valid_dataloader_2i_e, valid_dataloader_2i_c, test_dataloader_2i_e, test_dataloader_2i_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2i')
train_e_3i, train_c_3i, train_filter_e_3i, train_filter_c_3i, test_e_3i, test_c_3i, test_filter_e_3i, test_filter_c_3i, valid_dataloader_3i_e, valid_dataloader_3i_c, test_dataloader_3i_e, test_dataloader_3i_c = load_train_and_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='3i')
test_e_pi, test_c_pi, test_filter_e_pi, test_filter_c_pi, valid_dataloader_pi_e, valid_dataloader_pi_c, test_dataloader_pi_e, test_dataloader_pi_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='pi')
test_e_ip, test_c_ip, test_filter_e_ip, test_filter_c_ip, valid_dataloader_ip_e, valid_dataloader_ip_c, test_dataloader_ip_e, test_dataloader_ip_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='ip')
test_e_2u, test_c_2u, test_filter_e_2u, test_filter_c_2u, valid_dataloader_2u_e, valid_dataloader_2u_c, test_dataloader_2u_e, test_dataloader_2u_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='2u')
test_e_up, test_c_up, test_filter_e_up, test_filter_c_up, valid_dataloader_up_e, valid_dataloader_up_c, test_dataloader_up_e, test_dataloader_up_c = load_test_data(cfg.root + cfg.dataset + '/', cfg.num_workers, e_dict, c_dict, r_dict, query_type='up')

train_data = torch.cat([train_ot, train_is, train_e_1p, train_c_1p, train_e_2p, train_c_2p, train_e_3p, train_c_3p, train_e_2i, train_c_2i, train_e_3i, train_c_3i], dim=0)
train_filters_e = {'1p': train_filter_e_1p, '2p': train_filter_e_2p, '3p': train_filter_e_3p, '2i': train_filter_e_2i, '3i': train_filter_e_3i}
train_filters_c = {'1p': train_filter_c_1p, '2p': train_filter_c_2p, '3p': train_filter_c_3p, '2i': train_filter_c_2i, '3i': train_filter_c_3i}
train_dataset = TrainDataset(e_dict, c_dict, train_data, num_ng=cfg.num_ng, filters={'e': train_filters_e, 'c': train_filters_c})
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.bs, num_workers=cfg.num_workers, shuffle=True, drop_last=True)

device="cpu"
model.to(device)
model.eval()
print('Validating Entity Answering:')
# _, rr_1p_e, h1_1p_e, h3_1p_e, h10_1p_e, h50_1p_e = evaluate_e(model, valid_dataloader_1p_e,
#                                                               test_filter_e_1p, device, query_type='1p')
# _, rr_2p_e, h1_2p_e, h3_2p_e, h10_2p_e, h50_2p_e = evaluate_e(model, valid_dataloader_2p_e,
#                                                               test_filter_e_2p, device, query_type='2p')
# _, rr_3p_e, h1_3p_e, h3_3p_e, h10_3p_e, h50_3p_e = evaluate_e(model, valid_dataloader_3p_e,
#                                                               test_filter_e_3p, device, query_type='3p')
# _, rr_2i_e, h1_2i_e, h3_2i_e, h10_2i_e, h50_2i_e = evaluate_e(model, valid_dataloader_2i_e,
#                                                               test_filter_e_2i, device, query_type='2i')
# _, rr_3i_e, h1_3i_e, h3_3i_e, h10_3i_e, h50_3i_e = evaluate_e(model, valid_dataloader_3i_e,
#                                                               test_filter_e_3i, device, query_type='3i')
# _, rr_pi_e, h1_pi_e, h3_pi_e, h10_pi_e, h50_pi_e = evaluate_e(model, valid_dataloader_pi_e,
#                                                               test_filter_e_pi, device, query_type='pi')
_, rr_ip_e, h1_ip_e, h3_ip_e, h10_ip_e, h50_ip_e = evaluate_e(model, valid_dataloader_ip_e,
                                                              test_filter_e_ip, device, query_type='ip')
_, rr_2u_e, h1_2u_e, h3_2u_e, h10_2u_e, h50_2u_e = evaluate_e(model, valid_dataloader_2u_e,
                                                              test_filter_e_2u, device, query_type='2u')
_, rr_up_e, h1_up_e, h3_up_e, h10_up_e, h50_up_e = evaluate_e(model, valid_dataloader_up_e,
                                                              test_filter_e_up, device, query_type='up')
mrr_e = round(sum([rr_1p_e, rr_2p_e, rr_3p_e, rr_2i_e, rr_3i_e, rr_pi_e, rr_ip_e, rr_2u_e, rr_up_e]) / 9, 3)
mh1_e = round(sum([h1_1p_e, h1_2p_e, h1_3p_e, h1_2i_e, h1_3i_e, h1_pi_e, h1_ip_e, h1_2u_e, h1_up_e]) / 9, 3)
mh3_e = round(sum([h3_1p_e, h3_2p_e, h3_3p_e, h3_2i_e, h3_3i_e, h3_pi_e, h3_ip_e, h3_2u_e, h3_up_e]) / 9, 3)
mh10_e = round(
    sum([h10_1p_e, h10_2p_e, h10_3p_e, h10_2i_e, h10_3i_e, h10_pi_e, h10_ip_e, h10_2u_e, h10_up_e]) / 9, 3)
mh50_e = round(
    sum([h50_1p_e, h50_2p_e, h50_3p_e, h50_2i_e, h50_3i_e, h50_pi_e, h50_ip_e, h50_2u_e, h50_up_e]) / 9, 3)
print(f'Entity Answering Mean: \n MRR: {mrr_e}, H1: {mh1_e}, H3: {mh3_e}, H10: {mh10_e}, H50: {mh50_e}')