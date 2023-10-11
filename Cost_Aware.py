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

np.set_printoptions(suppress=True)
np.set_printoptions(precision=0)
torch.set_printoptions(profile="full")
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
import matplotlib.pyplot as plt
import csv
import time
from scipy import stats
from utils import *
torch.cuda.max_memory_allocated(16)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
__constants__ = ['ignore_index', 'reduction']
ignore_index: int




class BaseAttack(Module):

    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device
        self.modified_adj = None
        self.modified_features = None
        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """n_perturbations:  Number of edge removals/additions"""
        pass

    def check_adj(self, adj):
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.tocsr().max() == 1, "Max value should be 1!"
        assert adj.tocsr().min() == 0, "Min value should be 0!"

    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is symmetric, unweighted, all-zero diagonal.
        """
        assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj.diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"

    def save_adj(self, root=r'/tmp/', name='mod_adj'):

        name = name + '.npz'
        modified_adj = self.modified_adj
        if type(modified_adj) is torch.Tensor:
            sparse_adj = to_scipy(modified_adj)
            sp.save_npz(osp.join(root, name), sparse_adj)

    def save_features(self, root=r'/tmp/', name='mod_features'):

        assert self.modified_features is not None, \
            'modified_features is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_features = self.modified_features

        if type(modified_features) is torch.Tensor:
            sparse_features = to_scipy(modified_features)
            sp.save_npz(osp.join(root, name), sparse_features)
        else:
            sp.save_npz(osp.join(root, name), modified_features)


class BaseMeta(BaseAttack):
    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0, attack_structure=True,
                 attack_features=False, undirected=True, device='cpu'):
        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_
        self.modified_adj = None
        self.modified_features = None
        if attack_structure:
            self.undirected = undirected
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))  # main diagonal
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.size(0), 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, labels, idx_train):
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training

    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        t_d_min = torch.tensor(2.0).to(self.device)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.nnodes, self.nnodes)) - np.eye(self.nnodes)).nonzero()).T
        allowed_mask, current_ratio = likelihood_ratio_filter(t_possible_edges,
                                                              modified_adj,
                                                              ori_adj, t_d_min,
                                                              ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # 在这段代码中，modified_adj 是修改后的邻接矩阵，-2 * modified_adj + 1 这个操作在数值上是将1转变为-1，将0转变为1，也就是说原本连接的边（邻接矩阵中值为1）变为-1，
        # 原本未连接的边（邻接矩阵中值为0）变为1。这样做的目的是希望优化方向向着原本没有边的地方增加边，向着原本有边的地方减少边，也就是说，如果梯度为正，我们希望增加边，如果梯度为负，
        # 我们希望减少边，这符合梯度下降法的一般思想。换句话说，-2 * modified_adj + 1 这个操作提供了一个方向性的提示：对于原来存在的边，我们应该考虑是否移除它（使邻接矩阵的值从1变为0）
        # 以减小损失；对于原来不存在的边，我们应该考虑是否添加它（使邻接矩阵的值从0变为1）以减小损失。然后，通过将这个方向性的提示与原始的梯度 adj_grad 相乘，我们就得到了可以用来更新邻接矩阵的元梯度。
        # 在梯度下降优化方法中，我们希望通过逐渐改变模型的参数来最小化损失函数。这里的改变是按照梯度的方向进行的，也就是说，如果一个参数的梯度是正的，那么增加这个参数会导致损失增加；反之
        # 如果一个参数的梯度是负的，那么增加这个参数会导致损失减少。对于图结构数据来说，参数通常是图中的边，也就是邻接矩阵中的元素。当我们优化这种图结构数据的时候，我们也是希望通过增加或删除边来最小
        # 化损失函数。具体到你的问题，就是：如果一个边的梯度是正的，那么增加这个边（即将这个边从不存在变为存在，邻接矩阵中对应的元素从0变为1）会使得损失增加。因此，我们希望保持这个边不存在，即不增加这个边。
        # 如果一个边的梯度是负的，那么增加这个边会使得损失减少。因此，我们希望增加这个边。这样理解并不是绝对准确的，因为这里其实是在对邻接矩阵做微分，我们是在考虑在当前状态下微小的改变会对损失产生怎样的影响，而不是直接增加或删除边。但这种理解有助于把握梯度下降在图结构优化中的基本思想。
        # Make sure that the minimum entry is 0.
        adj_meta_grad -= adj_meta_grad.min()  # 这一行确保 adj_meta_grad 的最小值是 0，即所有的元梯度都是非负的。这可以防止在优化过程中对邻接矩阵进行无效的或者过于激进的修改。只有加边，没有删边
        # Filter self-loops
        adj_meta_grad -= torch.diag(
            torch.diag(adj_meta_grad, 0))  # 这一行消除了元梯度矩阵的对角线元素。在图的邻接矩阵中，对角线元素通常用于表示自环（self-loops）。这里的操作意味着我们不希望修改自环。
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad * singleton_mask

        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask.to(self.device)
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad


class CostAware(BaseMeta):
    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True,
                 device='cpu', with_bias=False, lambda_=0, train_iters=100, lr=0.1, momentum=0.9, weight1=1.0,
                 weight2=1.0, sigma1=1.0, sigma2=1.0):
        super(CostAware, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features,
                                        undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias
        self.weights = []
        self.bias = []
        self.w_velocities = []
        self.weight1 = weight1
        self.weight2 = weight2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass
        self.lossA = multi_objective_loss(device, weight1, weight2, sigma1, sigma2)
        self.lossB = similar_distributionode(device, weight1, weight2, sigma1, sigma2)
        self.lossD = NLL_margin_loss(device)
        self.lossE = similar_distribution(device, weight1, weight2, sigma1, sigma2)
        # self.loss = XE_LOSS(7,device)
        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)
            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)
            previous_size = nhid
        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)
        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def MarginLoss_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels,idx_train_label_6,
    idx_unlabeled_label_6,idx_test_label_6, labels_self_training):
        hidden = features

        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.softmax(hidden, dim=1)
        output1 = F.log_softmax(hidden, dim=1)

        attack_loss = labelloss(output1[idx_unlabeled_label_6], labels_self_training[idx_unlabeled_label_6])


        loss_test_val = F.nll_loss(output1[idx_unlabeled_label_6], labels[idx_unlabeled_label_6])
        # confidence_list = []
        # l2_norm_list = []
        #
        # for idx in idx_unlabeled:
        #     margin_node = classification_margin_all(output1[idx], labels[idx])
        #
        #     # preds_node = self.lossB(output[idx].unsqueeze(0), labels_self_training[idx].unsqueeze(0), margin_node)
        #
        #     preds_node = CWlossNode(output1[idx], labels_self_training[idx])
        #     confidence_list.append(margin_node)
        #     # confidence = output[idx, labels_self_training[idx]]
        #     if preds_node==0:
        #         l2_norm=torch.tensor(0)
        #
        #     elif preds_node<0:
        #         adj_grad = torch.autograd.grad(preds_node, self.adj_changes, retain_graph=True,allow_unused=True)[0]
        #
        #         l2_norm = torch.norm(adj_grad, p=2)
        #     else:
        #         l2_norm=torch.tensor(0)
        #         # CA
        #     l2_norm_np = l2_norm.cpu().detach().numpy()
        #         # confidence_list.append(confidence_np)
        #
        #     l2_norm_list.append(l2_norm_np)
        #
        # # 绘制散点图
        #
        # plt.figure(figsize=(5, 4))
        # plt.scatter(confidence_list, l2_norm_list, c='b', marker='o')
        # plt.xlim([min(confidence_list) - 0.05, max(confidence_list) + 0.05])
        # plt.ylim([min(l2_norm_list) - 0.05, max(l2_norm_list) + 0.05])
        # plt.xlabel("margin", fontsize=16)
        # plt.xticks(fontsize=16, rotation=90)
        # plt.yticks(fontsize=16)
        # plt.savefig('results_figure/cora_CWcost_l2.pdf', bbox_inches='tight', dpi=1000)
        # plt.show()
        print('GCN loss on unlabeled data:{}'.format(loss_test_val.item()))
        print('GCN acc on unlabeled data:{}'.format(accuracy(output1[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss:{}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        acc = accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()

        return adj_grad,feature_grad

    def Multi_objective_grad(self, features, adj_norm, idx_train, idx_unlabeled, idx_full,labels, labels_self_training,ori_adj):
        hidden = features

        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.softmax(hidden, dim=1)
        output1 = F.log_softmax(hidden, dim=1)

        margin_train= margin_compute(output[idx_train], labels[idx_train])
        score = structural_scores(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_labeled = self.lossA(output1[idx_train], labels[idx_train], margin_train,score)
        margin_unlabeled = margin_compute(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_unlabeled = self.lossA(output1[idx_unlabeled], labels_self_training[idx_unlabeled],margin_unlabeled,score)
        margin_full = margin_compute(output[idx_full], labels[idx_full])
        valid_nodes = torch.where((margin_full > 0) & (margin_full < 0.2))[0]
        if output[valid_nodes , labels_self_training[valid_nodes ]].mean() <0.5:
            valid_nodes = torch.where((margin_full > 0) & (margin_full < 0.6))[0]
        valid_edges = [(i, j) for i in valid_nodes for j in valid_nodes if i != j]
        self.adj_changes = self.adj_changes[[i for i, j in valid_edges], [j for i, j in valid_edges]]

        if output[valid_nodes, labels_self_training[valid_nodes]].mean() < 0.5:
            retained_adj_changes = self.adj_changes
            shape = ori_adj.shape

            # create a new tensor filled with zeros with the same shape as the original adjacency matrix
            new_adj_changes = torch.zeros(shape, device=self.adj_changes.device)

            # put the values from self.adj_changes into the appropriate positions of the new tensor
            new_adj_changes[[i for i, j in valid_edges], [j for i, j in valid_edges]] = self.adj_changes

            # now replace the original self.adj_changes with the new tensor
            self.adj_changes = torch.nn.Parameter(new_adj_changes)

        loss_test_val = F.nll_loss(output1[idx_unlabeled], labels[idx_unlabeled])
        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabeled data:{}'.format(loss_test_val.item()))
        print('GCN acc on unlabeled data:{}'.format(accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss:{}'.format(attack_loss.item()))
        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss,self.adj_changes , retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        acc = accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()
        return adj_grad, feature_grad
    # 在BaseMeta类初始化时，self.adj_changes是一个全零矩阵。这只是起始点，表示一开始没有对图结构进行任何修改。然而，self.adj_changes被定义为模型的一个Parameter，这意味着它是模型参数的一部分，并且会在模型的训练过程中被优化。换句话说，self.adj_changes的值在模型训练过程中会发生改变。
    #
    # 在你的代码中，你计算了attack_loss对self.adj_changes的梯度，然后使用这个梯度来更新self.adj_changes。这是通过梯度下降或其他优化算法实现的，即通过沿着梯度的负方向更新参数的值以最小化损失函数（或在你的攻击模型中，最大化损失函数）。
    #
    # 总的来说，尽管self.adj_changes最初是全零矩阵，但在模型训练过程中，它的值会根据梯度进行更新，以改变图结构，实现图的攻击。
    def NLL_plus(self,features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):
        hidden = features

        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)


        output = F.log_softmax(hidden, dim=1)

        # margin = margin_compute(output[idx_train], labels[idx_train])
        #
        # loss_labeled = self.lossB(output[idx_train], labels[idx_train], margin)
        margin = margin_compute(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_unlabeled = self.lossE(output[idx_unlabeled], labels_self_training[idx_unlabeled], margin)
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        confidence_list = []
        l2_norm_list = []

        for idx in idx_unlabeled:
            margin_node = classification_margin_all(output[idx], labels[idx])

            # preds_node = self.lossB(output[idx].unsqueeze(0), labels_self_training[idx].unsqueeze(0), margin_node)

            preds_node = self.lossB(output[idx], labels_self_training[idx],margin_node)

            # confidence = output[idx, labels_self_training[idx]]

            adj_grad = torch.autograd.grad(preds_node, self.adj_changes, retain_graph=True)[0]
            l2_norm = torch.norm(adj_grad, p=2)
            # CA
            l2_norm_np = l2_norm.cpu().detach().numpy()
            # confidence_list.append(confidence_np)
            confidence_list.append(margin_node)
            l2_norm_list.append(l2_norm_np)

        # 绘制散点图

        plt.figure(figsize=(5, 4))
        plt.scatter(confidence_list, l2_norm_list, c='b', marker='o')
        plt.xlim([min(confidence_list)-0.05,max(confidence_list)+0.05])
        plt.ylim([min(l2_norm_list) - 0.05, max(l2_norm_list) + 0.05])
        plt.xlabel("margin", fontsize=16)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16)
        plt.savefig('results_figure/cora_costaware_l2.pdf',bbox_inches='tight', dpi=1000)
        plt.show()
        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabeled data:{}'.format(loss_test_val.item()))
        print('GCN acc on unlabeled data:{}'.format(accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss:{}'.format(attack_loss.item()))
        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        acc = accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()
        return adj_grad, feature_grad
    def weighted_CW(self,features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):
        hidden = features

        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)


        output = F.log_softmax(hidden, dim=1)
        output1 = F.softmax(hidden,dim=1)
        # margin = margin_compute(output[idx_train], labels[idx_train])
        #
        # loss_labeled = self.lossB(output[idx_train], labels[idx_train], margin)
        margin = margin_compute(output1[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_unlabeled = weighted_CWloss(output[idx_unlabeled], labels_self_training[idx_unlabeled], margin,self.weight1,self.sigma1,self.device)
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        confidence_list = []
        l2_norm_list = []

        for idx in idx_unlabeled:
            margin_node = classification_margin_all(output[idx], labels[idx])

            preds_node = CostCWNode(output[idx], labels_self_training[idx], margin_node,self.weight1,self.sigma1)
        #

            confidence_list.append(margin_node)
            # confidence = output[idx, labels_self_training[idx]]
            if preds_node == 0:
                l2_norm = torch.tensor(0)

            elif preds_node < 0:
                adj_grad = torch.autograd.grad(preds_node, self.adj_changes, retain_graph=True, allow_unused=True)[0]
                l2_norm = torch.norm(adj_grad, p=2)
            else:
                l2_norm = torch.tensor(0)
                # CA
            l2_norm_np = l2_norm.cpu().detach().numpy()
            # confidence_list.append(confidence_np)

            l2_norm_list.append(l2_norm_np)

            # 绘制散点图

        plt.figure(figsize=(5, 4))
        plt.scatter(confidence_list, l2_norm_list, c='b', marker='o')
        plt.xlim([min(confidence_list) - 0.05, max(confidence_list) + 0.05])
        plt.ylim([min(l2_norm_list) - 0.05, max(l2_norm_list) + 0.05])
        plt.xlabel("margin", fontsize=16)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16)
        plt.savefig('results_figure/cora_WCWcost_l2.pdf', bbox_inches='tight', dpi=1000)
        plt.show()
        print('GCN loss on unlabeled data:{}'.format(loss_test_val.item()))
        print('GCN acc on unlabeled data:{}'.format(accuracy(output1[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss:{}'.format(attack_loss.item()))

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabeled data:{}'.format(loss_test_val.item()))
        print('GCN acc on unlabeled data:{}'.format(accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss:{}'.format(attack_loss.item()))
        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        acc = accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()
        return adj_grad, feature_grad
    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled,idx_full,idx_train_label_6,
    idx_unlabeled_label_6,idx_test_label_6, n_perturbations, ll_constraint=True,
               ll_cutoff=0.004,attack_type = None):
        modified_edges_mask = torch.ones_like(ori_adj, dtype=torch.bool).to(self.device)

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            if attack_type == 'Margin':
                adj_grad, feature_grad = self.MarginLoss_grad(modified_features, adj_norm, idx_train,
                                                              idx_unlabeled,
                                                              labels,idx_train_label_6,
    idx_unlabeled_label_6,idx_test_label_6,
                                                              labels_self_training)
            if attack_type == 'Multi_objective':
                adj_grad, feature_grad = self.Multi_objective_grad(modified_features, adj_norm, idx_train,
                                                                   idx_unlabeled,idx_full,
                                                                   labels,
                                                                   labels_self_training,ori_adj)
            if attack_type == 'NLLmargin':
                adj_grad, feature_grad = self.NLL_plus(modified_features, adj_norm, idx_train,
                                                             idx_unlabeled,
                                                             labels,
                                                             labels_self_training)
            if attack_type =='weighted_CW':
                adj_grad, feature_grad = self.weighted_CW(modified_features, adj_norm, idx_train,
                                                       idx_unlabeled,
                                                       labels,
                                                       labels_self_training)
            with torch.no_grad():

                adj_meta_score = torch.tensor(0.0).to(self.device)
                # adj_meta_score1 = torch.tensor(0.0).to(self.device)
                feature_meta_score = torch.tensor(0.0).to(self.device)
                if self.attack_structure:
                    adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
                    adj_meta_score = adj_meta_score * modified_edges_mask  # apply the mask
                    # adj_meta_score1 = self.get_adj_score(adj_grad1, modified_adj, ori_adj, ll_constraint, ll_cutoff)

                if self.attack_features:
                    feature_meta_score = self.get_feature_score(features_grad, modified_features)
                # score_sum = adj_meta_score + adj_meta_score1
                if adj_meta_score.max() >= feature_meta_score.max():
                    # if score_sum.max() >= feature_meta_score.max():
                    adj_meta_argmax = torch.argmax(adj_meta_score)
                    # adj_meta_argmax = torch.argmax(score_sum)
                    row_idx, col_idx = unravel_index(adj_meta_argmax, ori_adj.shape)
                    self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1) # 1 和-1的矩阵
                    modified_edges_mask[row_idx][col_idx] = False
                    if self.undirected:
                        self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                        modified_edges_mask[col_idx][row_idx] = False
                else:
                    feature_meta_argmax = torch.argmax(feature_meta_score)
                    row_idx, col_idx = unravel_index(feature_meta_argmax, ori_features.shape)
                    self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)
                # 这一段是针对无向图的处理。在无向图中，边是双向的，即从row_idx到col_idx和从col_idx到row_idx的边都是存在的。所以如果图是无向的，
                # 那么我们需要在邻接矩阵中同时更新这两个方向的边。这就是为什么在这个if语句中，我们对邻接矩阵进行相同的更新，但方向是反过来的。
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()
