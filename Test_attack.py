import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import logging
import math
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from tqdm import tqdm
import torch.nn as nn
import os
import pandas as pd
from graphsage import *
np.set_printoptions(suppress=True)
np.set_printoptions(precision=0)
torch.set_printoptions(profile="full")
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import GAT

from deeprobust.graph.data import Dataset, Dpr2Pyg
import argparse
import matplotlib.pyplot as plt
import csv
import time
from scipy import stats
from utils import *
from weighted_GraD import *
from Cost_Aware import *
from Metattack import *
from Meta_tanh import *
torch.cuda.max_memory_allocated(16)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
__constants__ = ['ignore_index', 'reduction']
ignore_index: int
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=8
                    , help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['citeseer', 'polblogs','cora'],help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
parser.add_argument('--model', type=str, default='MetaSelf',
                    choices=['GraD', 'CostAware','MetaSelf','Sigmoid','Tanh'], help='model variant')
parser.add_argument('--weight1', type=float, default=4.5, help='loss weight')
parser.add_argument('--weight2', type=float, default=1, help='loss weight')
parser.add_argument('--sigma1', type=float, default=1, help='normal_changes')
parser.add_argument('--sigma2', type=float, default=1, help='normal_changes')
parser.add_argument('--number', type=float, default=1.0, help='iteration')
parser.add_argument('--GPU', type=int, default=0, choices=[8, 9,0,1,2,3,4,5,6,7,1],help='iteration')
parser.add_argument('--csv', type=str, default='margin_Meta_{}_{}_{}.csv',
                    choices=['margin_Meta_{}_{}_{}.csv','Sigmoid_{}_{}_{}.csv', 'margin_Grad_{}_{}_{}.csv', 'weighted_Grad_{}_{}_{}.csv','cost_{}_{}_{}.csv','costCom_{}_{}_{}.csv','Tanh_{}_{}_{}.csv','margin_cost_{}_{}_{}','Wmargin_cost_{}_{}_{}'], help='margin plot')

parser.add_argument('--attack_type', type=str, default='Meta',
                    choices=['Margin', 'Multi_objective', 'NLLmargin', 'Weighted_GraD','GraD','Meta','Tanh','Sigmoid','weighted_CW'], help='loss function')
args = parser.parse_args()
if torch.cuda.is_available():
    if hasattr(args, 'GPU') and args.GPU is not None:
        device = torch.device(f"cuda:{args.GPU}")
    else:
        device = torch.device("cuda:0")  # 默认选择第一个GPU
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)


def test(adj,gnn, acc_total, account_total1, account_total2, account_total3, account_total4, account_total5, account_total6,account_total7,
account_total8, account_total9, account_total10, GAP1, GAP2, features, labels, idx_train, idx_test,idx_full,idx_train_label_6,
    idx_unlabeled_label_6,idx_test_label_6):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    range_0_02 = 0
    range_02_04 = 0
    range_04_06 = 0
    range_06_08 = 0
    range_08_1 = 0
    range_neg02_0 = 0
    range_neg04_neg02 = 0
    range_neg06_neg04 = 0
    range_neg08_neg06 = 0
    range_neg1_neg08 = 0
    if gnn =='GCN':
        print('==Here are the results on GCN')
        GNN = GCN(nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=labels.max().item() + 1,
                  dropout=args.dropout, device=device)
        GNN = GNN.to(device)
        # GNN.fit(features, adj, labels, idx_train)
        GNN.fit(features, adj, labels, idx_train)
        output = GNN.output.cpu()
        output1 = torch.exp(output)
    if gnn == 'GraphSage':
        adj = adj.to(device)
        features = features.to(device)
        labels = labels.to(device)
        print('==Here are the results on Graphsage')
        GNN = GraphSage(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout)

        GNN = GNN.to(device)
        optimizer = optim.Adam(GNN.parameters(),
                               lr=args.lr, weight_decay=5e-4)
        GNN.train()

        for epoch in range(args.epochs):
            optimizer.zero_grad()
            output = GNN(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # loss_train = F.nll_loss(output[idx_train_label_6], labels[idx_train_label_6])
            loss_train.backward()
            optimizer.step()

        GNN.eval()
        output = GNN(features, adj).to(device)
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
        output1 = torch.exp(output)
    if gnn =='GAT':
        print('==Here are the results on GAT')
        GNN = GAT(nfeat=features.shape[1],
                  nhid=args.hidden, heads=8,
                  nclass=labels.max().item() + 1,
                  dropout=args.dropout, device=device)

        data = Dataset(root='/tmp/', name=args.dataset)
        pyg_data = Dpr2Pyg(data)  # convert deeprobust dataset to pyg dataset

        GNN = GNN.to(device)
        GNN.fit(pyg_data, patience=100, verbose=True)
        output = GNN.output.cpu()
        output1 = torch.exp(output)
    margin_all = margin_compute(output1[idx_test], labels[idx_test])
    margin_all_np = margin_all.detach().cpu().numpy()

    # 保存到CSV文件，假设你要保存的文件名是'margin_all.csv'
    path = os.path.join('exper_log/margin', args.csv.format(args.dataset, args.ptb_rate, gnn))
    np.savetxt(path, margin_all_np, delimiter=',')

    # np.savetxt(args.txt.format(args.dataset, args.ptb_rate), margin_all.detach().numpy(), fmt='%f')
    margin_dict = {}

    for idx in idx_full:
        margin = classification_margin_all(output[idx], labels[idx])
        if margin < 0:  # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
        if margin < 0.2:
            count1 = count1 + 1
        if margin >= 0.2 and margin < 0.4:
            count2 = count2 + 1
        if margin >= 0.4 and margin < 0.6:
            count3 = count3 + 1
        if margin >= 0.6 and margin < 0.8:
            count4 = count4 + 1
        if margin >= 0.8:
            count5 = count5 + 1
    GAPp = sum(margin_dict.values()) / len(margin_dict)

    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)

    high = [x for x, y in sorted_margins[-10:]]

    count6 = 0
    count7 = 0
    count8 = 0
    count9 = 0
    count10 = 0
    margin_dict_suc = {}
    # print(len(idx_test))
    for idx in idx_full:
        margin = classification_margin_all(output[idx], labels[idx])
        if margin > 0:  # only keep the nodes correctly classified
            continue
        margin_dict_suc[idx] = margin
        if margin >= -0.2:
            count6 = count6 + 1
        if margin >= -0.4 and margin < -0.2:
            count7 = count7 + 1
        if margin >= -0.6 and margin < -0.4:
            count8 = count8 + 1
        if margin >= -0.8 and margin < -0.6:
            count9 = count9 + 1
        if margin < -0.8:
            count10 = count10 + 1
        if margin == 0:
            print(margin)
    GAPn = sum(margin_dict_suc.values()) / len(margin_dict_suc)
    sorted_margins_suc = sorted(margin_dict_suc.items(), key=lambda x: x[1], reverse=True)
    margin_edges = margin_compute(output1[idx_full], labels[idx_full])
    for margin, change in zip(margin_edges, changes):
        if margin >= 0 and margin < 0.2:
            range_0_02+=change
        elif margin >= 0.2 and margin < 0.4:
            range_02_04 += change
        elif margin >= 0.4 and margin < 0.6:
            range_04_06 += change
        elif margin >= 0.6 and margin < 0.8:
            range_06_08 += change
        elif margin >= 0.8 and margin <= 1.0:
            range_08_1 += change
        elif margin >= -0.2 and margin < 0:
            range_neg02_0+= change
        elif margin >= -0.4 and margin < -0.2:
            range_neg04_neg02+= change
        elif margin >= -0.6 and margin < -0.4:
            range_neg06_neg04 += change
        elif margin >= -0.8 and margin < -0.6:
           range_neg08_neg06+= change
        elif margin >= -1.0 and margin < -0.8:
            range_neg1_neg08+= change
    print("edges in range 0-0.2:", range_0_02)
    print("edges in range 0.2-0.4:", range_02_04)
    print("edges in range 0.4-0.6:", range_04_06)
    print("edges in range 0.6-0.8:", range_06_08)
    print("edges in range 0.8-1.0:", range_08_1)
    print("edges in range -0.2 to 0:", range_neg02_0)
    print("edges in range -0.4 to -0.2:", range_neg04_neg02)
    print("edges in range -0.6 to -0.4:", range_neg06_neg04)
    print("edges in range -0.8 to -0.6:", range_neg08_neg06)
    print("edges in range -1 to -0.8:", range_neg1_neg08)
    loss_test = F.nll_loss(output[idx_test_label_6], labels[idx_test_label_6])
    acc_test = accuracy(output[idx_test_label_6], labels[idx_test_label_6])
    acc_total.append(acc_test.item())
    GAP1.append(GAPp)
    GAP2.append(GAPn)
    account_total1.append(count1)
    account_total2.append(count2)
    account_total3.append(count3)
    account_total4.append(count4)
    account_total5.append(count5)
    account_total6.append(count6)
    account_total7.append(count7)
    account_total8.append(count8)
    account_total9.append(count9)
    account_total10.append(count10)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


def main():
    GAP1 = []
    GAP2 = []
    GAP3 = []
    GAP4 = []
    acc_total0 = []
    acc_total1 = []
    account_total1 = []
    account_total2 = []
    account_total3 = []
    account_total4 = []
    account_total5 = []
    account_total6 = []
    account_total7 = []
    account_total8 = []
    account_total9 = []
    account_total10 = []
    account_total11 = []
    account_total12 = []
    account_total13 = []
    account_total14 = []
    account_total15 = []
    account_total16 = []
    account_total17 = []
    account_total18 = []
    account_total19 = []
    account_total20 = []

    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    idx_train_label_6 = idx_train[labels[idx_train]==6]
    idx_unlabeled_label_6 = idx_unlabeled[labels[idx_unlabeled]==6]
    idx_test_label_6 = idx_test[labels[idx_test]==6]
    idx_full = np.union1d(idx_train, idx_unlabeled)
    perturbations = int(args.ptb_rate * (adj.sum() // 2))
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)

    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    # if run == 1:
    #     name = 'clean_adj_polblogs'
    #     name = name + '.npz'
    #     root = r'/tmp/'
    #     if type(adj) is torch.Tensor:
    #         sparse_adj = to_scipy(adj)
    #         sp.save_npz(osp.join(root, name), sparse_adj)
    #     np.savetxt(f'clean_adj_polblogs', adj, fmt='%.00f')
    #Setup Attack Model
    if args.model == 'MetaSelf':
        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                          attack_features=False, device=device, lambda_=0)
    if args.model == 'GraD':
        model = GraD(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                     attack_features=False, device=device, lambda_=0)

    if args.model == 'CostAware':
        model = CostAware(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                          attack_features=False, device=device, lambda_=0, weight1=args.weight1, weight2=args.weight2,
                          sigma1=args.sigma1, sigma2=args.sigma2)
    if args.model == 'Sigmoid':
        model = Tanh(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                     attack_features=False, device=device, lambda_=0)
    if args.model == 'Tanh':
        model = Tanh(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                     attack_features=False, device=device, lambda_=0)

    model = model.to(device)
    start_time = time.time()

    model.attack(features, adj, labels, idx_train, idx_unlabeled,idx_full, perturbations, ll_constraint=True,ll_cutoff = 0.004,attack_type = args.attack_type)
    model.save_adj(root='exper_log', name=f'mod_adj')
    model.save_features(root='exper_log', name='mod_features')
    elapsed_time = time.time() - start_time
    print(f"Attack Code executed in: {elapsed_time:.2f} seconds")
    modified_adj = model.modified_adj
    modified_adj = modified_adj.detach()
    with open("./exper_log/log/execution_log.txt", "a") as log_file:
        log_file.write(f'The attack_type is {args.attack_type},the data is {args.dataset}\n ')
        log_file.write(f"Code executed in: {elapsed_time:.2f} seconds\n")
    gnns = ['GCN', 'GraphSage','GAT']
    # gnns =['GCN']
    # gnns = ['GraphSage']
    for gnn in gnns:
        path = os.path.join('exper_log/model', args.csv.format(args.dataset, args.ptb_rate, args.attack_type, gnn))
        np.savetxt(path, modified_adj.cpu().numpy())

        modified_adj = np.loadtxt(path, delimiter=' ')
        # changes = torch.sum(torch.abs(modified_adj - adj), dim=1)
        modified_adj = torch.FloatTensor(modified_adj).cuda()
        for seed in range(1, 11):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device != 'cpu':
               torch.cuda.manual_seed(seed)
            print('Running the {}-th time'.format(seed))
            print("clean graph test")
            test(adj, gnn, acc_total0, account_total1, account_total2, account_total3, account_total4, account_total5,
             account_total6, account_total7,
             account_total8, account_total9, account_total10, GAP1, GAP2, features, labels, idx_train, idx_test,idx_full)
    #         test(adj, gnn, acc_total0, account_total1, account_total2, account_total3, account_total4, account_total5,
    #              account_total6, account_total7,
    #              account_total8, account_total9, account_total10, GAP1, GAP2, features, labels, idx_train, idx_test,idx_full,idx_train_label_6,
    # idx_unlabeled_label_6,idx_test_label_6)
            print("final modified_adj")
            test(modified_adj, gnn, acc_total1, account_total11, account_total12, account_total13, account_total14,
                 account_total15,
                 account_total16, account_total17,
                 account_total18, account_total19, account_total20, GAP3, GAP4, features, labels, idx_train, idx_test,
                 idx_full)
    #         test(modified_adj, gnn, acc_total1, account_total11, account_total12, account_total13, account_total14,
    #              account_total15,
    #              account_total16, account_total17,
    #              account_total18, account_total19, account_total20, GAP3, GAP4, features, labels, idx_train, idx_test,idx_full,idx_train_label_6,
    # idx_unlabeled_label_6,idx_test_label_6)
    #
        accuracy1 = np.mean(acc_total0)
        accuracy2 = np.mean(acc_total1)
        interval1, interval2 = stats.t.interval(0.95, 9, accuracy1,
                                                np.std(acc_total0, ddof=1) / np.sqrt(len(acc_total0)))
        interval3, interval4 = stats.t.interval(0.95, 9, accuracy2,
                                                np.std(acc_total1, ddof=1) / np.sqrt(len(acc_total1)))
        print("confidence of clean adj: %.4f"% ((interval2 - interval1) / 2))
        print("confidence: %.4f"% ((interval4 - interval3) / 2))
        print("Mean Accuracy of clean:%.4f" % np.mean(acc_total0))
        print("standard Deviation of clean:%.4f" % np.std(acc_total0, ddof=1))
        print("Mean Accuracy:%.4f" % np.mean(acc_total1))
        with open('{}_{}_{}.csv'.format(args.dataset,args.ptb_rate,args.attack_type), 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([args.number])
            writer.writerow([accuracy2])
        print("standard Deviation:%.4f" % np.std(acc_total1, ddof=1))
        logging.basicConfig(filename=os.path.join('exper_log/log','{}_{}_{}_{}.log'.format(args.attack_type,args.dataset, args.ptb_rate,gnn)), level=logging.INFO)

        logging.info("confidence of clean adj: %.4f "%((interval2 - interval1) / 2))
        logging.info("confidence: %.4f"%((interval4 - interval3) / 2))
        logging.info("Mean Accuracy of clean: %.4f"% np.mean(acc_total0))
        logging.info("standard Deviation of clean: %.4f"% np.std(acc_total0, ddof=1))
        logging.info("Mean Accuracy: %.4f"% np.mean(acc_total1))
        logging.info("standard Deviation:%.4f" % np.std(acc_total1, ddof=1))
        parameter_titles = ['number', 'Mean Accuracy of clean', 'std', 'Mean Accuracy', 'std', 'Mean GAP pf clean positive',
                            'std',
                            'Mean GAP pf clean negative', 'std', 'Mean GAP pf perturbed positive', 'std',
                            'Mean GAP pf perturbed negative', 'std']
        parameter_values = [args.number, np.mean(acc_total0), (interval2 - interval1) / 2, np.mean(acc_total1),
                            (interval4 - interval3) / 2, np.mean(GAP1), np.std(GAP1, ddof=1), np.mean(GAP2),
                            np.std(GAP2, ddof=1), np.mean(GAP3), np.std(GAP3, ddof=1), np.mean(GAP4), np.std(GAP4, ddof=1)]
        with open('exper_log/accuracy1.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metattack'])
            writer.writerow(parameter_titles)

            # 继续写入数据行
            writer.writerow(parameter_values)
        parameter_titles1 = ['number', 'p1', 'p2', 'p3', 'p4', 'p5', 'n1', 'n2', 'n3', 'n4', 'n5']
        parameter_values1 = [args.number, int(np.mean(account_total1)), int(np.mean(account_total2)),
                             int(np.mean(account_total3)),
                             int(np.mean(account_total4)),
                             int(np.mean(account_total5)), int(np.mean(account_total6)), int(np.mean(account_total7)),
                             int(np.mean(account_total8)),
                             int(np.mean(account_total9)), int(np.mean(account_total10))]
        parameter_values2 = [args.number, int(np.mean(account_total11)), int(np.mean(account_total12)),
                             int(np.mean(account_total13)),
                             int(np.mean(account_total14)),
                             int(np.mean(account_total15)), int(np.mean(account_total16)), int(np.mean(account_total17)),
                             int(np.mean(account_total18)),
                             int(np.mean(account_total19)), int(np.mean(account_total20))]
        with open('exper_log/accuracy1_1.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metattack'])
            writer.writerow(parameter_titles1)

            # 继续写入数据行
            writer.writerow(parameter_values1)
            writer.writerow(parameter_values2)
        print("Mean GAP1:%.4f" % np.mean(GAP1))
        print("standard Deviation:%.4f" % np.std(GAP1, ddof=1))
        print("Mean GAP2:%.4f" % np.mean(GAP2))
        print("standard Deviation:%.4f" % np.std(GAP2, ddof=1))
        print("Mean GAP3:%.4f" % np.mean(GAP3))
        print("standard Deviation:%.4f" % np.std(GAP3, ddof=1))
        print("Mean GAP4:%.4f" % np.mean(GAP4))
        print("standard Deviation:%.4f" % np.std(GAP4, ddof=1))
        logging.info("standard Deviation: %.4f" % np.std(acc_total1, ddof=1))
        logging.info("Mean GAP1: %.4f" % np.mean(GAP1))
        logging.info("standard Deviation: %.4f" % np.std(GAP1, ddof=1))
        logging.info("Mean GAP2: %.4f" % np.mean(GAP2))
        logging.info("standard Deviation: %.4f" % np.std(GAP2, ddof=1))
        logging.info("Mean GAP3: %.4f" % np.mean(GAP3))
        logging.info("standard Deviation: %.4f" % np.std(GAP3, ddof=1))
        logging.info("Mean GAP4: %.4f" % np.mean(GAP4))
        logging.info("standard Deviation: %.4f" % np.std(GAP4, ddof=1))
        print("positive margins", int(np.mean(account_total1)), int(np.mean(account_total2)), int(np.mean(account_total3)),
              int(np.mean(account_total4)),
              int(np.mean(account_total5)))
        print("negative margins", int(np.mean(account_total6)), int(np.mean(account_total7)), int(np.mean(account_total8)),
              int(np.mean(account_total9)), int(np.mean(account_total10)))
        print("positive margins", int(np.mean(account_total11)), int(np.mean(account_total12)),
              int(np.mean(account_total13)),
              int(np.mean(account_total14)),
              int(np.mean(account_total15)))
        print("negative margins", int(np.mean(account_total16)), int(np.mean(account_total17)),
              int(np.mean(account_total18)),
              int(np.mean(account_total19)), int(np.mean(account_total20)))
        logging.info("==clean graph")
        logging.info("positive margins: %d, %d, %d, %d, %d" % (
            int(np.mean(account_total1)),
            int(np.mean(account_total2)),
            int(np.mean(account_total3)),
            int(np.mean(account_total4)),
            int(np.mean(account_total5))))

        logging.info("negative margins: %d, %d, %d, %d, %d" % (
            int(np.mean(account_total6)),
            int(np.mean(account_total7)),
            int(np.mean(account_total8)),
            int(np.mean(account_total9)),
            int(np.mean(account_total10))))
        logging.info("==modified graph")
        logging.info("positive margins: %d, %d, %d, %d, %d" % (
            int(np.mean(account_total11)),
            int(np.mean(account_total12)),
            int(np.mean(account_total13)),
            int(np.mean(account_total14)),
            int(np.mean(account_total15))))

        logging.info("negative margins: %d, %d, %d, %d, %d" % (
            int(np.mean(account_total16)),
            int(np.mean(account_total17)),
            int(np.mean(account_total18)),
            int(np.mean(account_total19)),
            int(np.mean(account_total20))))

        print(
            'The percent of nodes with gap >0.8 is: {:.2%}'.format(int(np.mean(account_total15)) / (
                    int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
                np.mean(account_total11)) + int(np.mean(account_total15)))))
        # # if you want to save the modified adj/features, uncomment the code below
        # model.save_adj(root='./', name=f'mod_adj')
        # model.save_features(root='./', name='mod_features')
        print(
            'The percent of nodes with gap >0.6<0.8 is: {:.2%}'.format(int(np.mean(account_total14)) / (
                    int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
                np.mean(account_total11)) + int(np.mean(account_total15)))))
        print(
            'The percent of nodes with gap >0.4<0.6 is: {:.2%}'.format(int(np.mean(account_total13)) / (
                    int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
                np.mean(account_total11)) + int(np.mean(account_total15)))))
        print('The percent of nodes with gap >0.2<0.4 is: {:.2%}'.format(int(np.mean(account_total12)) / (
                int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
            np.mean(account_total11)) + int(np.mean(account_total15)))))
        print('The percent of nodes with gap <0.2 is: {:.2%}'.format(int(np.mean(account_total11)) / (
                int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
            np.mean(account_total11)) + int(np.mean(account_total11)))))
        print(
            'The percent of nodes with gap <-0.8 is: {:.2%}'.format(int(np.mean(account_total20)) / (
                    int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
                np.mean(account_total16)) + int(np.mean(account_total20)))))
        print(
            'The percent of nodes with gap >-0.8<-0.6 is :{:.2%}'.format(int(np.mean(account_total19)) / (
                    int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
                np.mean(account_total16)) + int(np.mean(account_total20)))))
        print(
            'The percent of nodes with gap >-0.6<-0.4 is: {:.2%}'.format(int(np.mean(account_total18)) / (
                    int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
                np.mean(account_total16)) + int(np.mean(account_total20)))))
        print(
            'The percent of nodes with gap >-0.4<-0.2 is: {:.2%}'.format(int(np.mean(account_total17)) / (
                    int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
                np.mean(account_total16)) + int(np.mean(account_total20)))))
        print('The percent of nodes with gap >-0.2 is: {:.2%}'.format(int(np.mean(account_total16)) / (
                int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
            np.mean(account_total16)) + int(np.mean(account_total20)))))
        logging.info('The percent of nodes with gap >0.8 is: {:.2%}'.format(int(np.mean(account_total15)) / (
                int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
            np.mean(account_total11)) + int(np.mean(account_total15)))))

        logging.info('The percent of nodes with gap >0.6<0.8 is: {:.2%}'.format(int(np.mean(account_total14)) / (
                int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
            np.mean(account_total11)) + int(np.mean(account_total15)))))
        logging.info('The percent of nodes with gap >0.4<0.6 is: {:.2%}'.format(int(np.mean(account_total13)) / (
                int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
            np.mean(account_total11)) + int(np.mean(account_total15)))))
        logging.info('The percent of nodes with gap >0.2<0.4 is: {:.2%}'.format(int(np.mean(account_total12)) / (
                int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
            np.mean(account_total11)) + int(np.mean(account_total15)))))
        logging.info('The percent of nodes with gap <0.2 is: {:.2%}'.format(int(np.mean(account_total11)) / (
                int(np.mean(account_total14)) + int(np.mean(account_total13)) + int(np.mean(account_total12)) + int(
            np.mean(account_total11)) + int(np.mean(account_total11)))))
        logging.info(
            'The percent of nodes with gap <-0.8 is: {:.2%}'.format(int(np.mean(account_total20)) / (
                    int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
                np.mean(account_total16)) + int(np.mean(account_total20)))))
        logging.info(
            'The percent of nodes with gap >-0.8<-0.6 is :{:.2%}'.format(int(np.mean(account_total19)) / (
                    int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
                np.mean(account_total16)) + int(np.mean(account_total20)))))
        logging.info(
            'The percent of nodes with gap >-0.6<-0.4 is: {:.2%}'.format(int(np.mean(account_total18)) / (
                    int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
                np.mean(account_total16)) + int(np.mean(account_total20)))))
        logging.info(
            'The percent of nodes with gap >-0.4<-0.2 is: {:.2%}'.format(int(np.mean(account_total17)) / (
                    int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
                np.mean(account_total16)) + int(np.mean(account_total20)))))
        logging.info('The percent of nodes with gap >-0.2 is: {:.2%}'.format(int(np.mean(account_total16)) / (
                int(np.mean(account_total19)) + int(np.mean(account_total18)) + int(np.mean(account_total17)) + int(
            np.mean(account_total16)) + int(np.mean(account_total20)))))


if __name__ == '__main__':
    main()
