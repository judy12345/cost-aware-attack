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
import sys
import matplotlib.pyplot as plt

import pickle as pkl
import networkx as nx
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
torch.cuda.max_memory_allocated(16)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
__constants__ = ['ignore_index', 'reduction']
ignore_index: int
def f(x, alpha, beta):
    x = torch.tensor(x)
    beta = torch.tensor(beta)
    return alpha * torch.exp(-beta * x ** 2)
# x = torch.linspace(-1, 1, 1000)  # Adjust the range of x values as needed
#
# # Apply the function for the given conditions
# y = torch.where(x < 0, f(x, 1, 1), f(x, 4.5, 1))
#
# # # Plot the function
# plt.figure(figsize=(8, 6))
# plt.plot(x.numpy(), y.numpy(), label=r'$w(v)$')
#
#
# plt.xlabel('margin', fontsize=24)
# plt.ylabel('w(v)', fontsize=24)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(True)
# plt.legend(fontsize=16)
# plt.tight_layout()
#
# # Save the plot as a PDF with 300 DPI
# plt.savefig("function_plot.pdf", dpi=300)
#
# plt.show()



def delete(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

class NLL_margin_loss(torch.nn.Module):
    def __init__(self, device):
        super( NLL_margin_loss, self).__init__()

        self.m_device = device

    def forward(self, preds,lables,score):
        indices = torch.nonzero(score < 0).squeeze()
        weight_margin = 1  # choose an appropriate value for this
        margin_loss = weight_margin * (torch.sum(score[indices]) / len(indices))
        NLL = F.nll_loss(preds,labels)
        return margin_loss

class structure_loss(torch.nn.Module):
    def __init__(self, item_num, device):
        super(structure_loss, self).__init__()
        self.m_item_num = item_num
        self.m_device = device

    def forward(self, score):
        # indices = torch.nonzero(score < 0).squeeze()
        weight_margin = 1  # choose an appropriate value for this
        margin_loss = weight_margin * (torch.sum(score) / len(score))
        return margin_loss
class Grad_DynamicMarginLossNode(torch.nn.Module):
    def __init__(self, device, weight1, weight2, sigma1, sigma2):
        super(Grad_DynamicMarginLossNode, self).__init__()
        self.m_device = device
        self.weight1 = weight1
        self.weight2 = weight2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def forward(self, preds, margin):
        with torch.no_grad():
            margin = torch.tensor(margin, device=self.m_device)
            if margin>0:
                w = f(margin, self.weight1, self.sigma1)
            if margin<0:
                w = f(margin, self.weight2, self.sigma2)
        loss = -w*preds

            # margin = margin.clone().detach().requires_grad_(True)
        #     positive_margin_indices = torch.nonzero(margin > 0).squeeze()
        #     negative_margin_indices = torch.nonzero(margin < 0).squeeze()
        #     filtered_margin_tensor = torch.zeros(margin.shape[0], dtype=torch.float32, device=self.m_device)
        #     filtered_margin_tensor[positive_margin_indices] = margin[positive_margin_indices]
        #     filtered_margin_tensor[negative_margin_indices] = 0.0
        #
        #     filtered_margin_tensor_neg = torch.zeros(margin.shape[0], dtype=torch.float32, device=self.m_device)
        #     filtered_margin_tensor_neg[negative_margin_indices] = margin[negative_margin_indices]
        #     filtered_margin_tensor_neg[positive_margin_indices] = 0.0
        #     w1 = f(filtered_margin_tensor, self.weight1, self.sigma1)
        #     w2 = f(filtered_margin_tensor_neg, self.weight2, self.sigma2)
        # N_positive_margin = preds[positive_margin_indices]
        # filtered_N1 = torch.zeros(margin.shape[0], dtype=preds.dtype, device=self.m_device)
        # filtered_N1[positive_margin_indices] = N_positive_margin
        # N_negative_margin = preds[negative_margin_indices]
        # filtered_N = torch.zeros(margin.shape[0], dtype=preds.dtype, device=self.m_device)
        # filtered_N[negative_margin_indices] = N_negative_margin
        #
        # loss1 = torch.sum(
        #     w1 * filtered_N1)
        # loss2 = torch.sum(
        #     w2 * filtered_N)
        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)
        # loss = -(loss1 + loss2) / len(margin)
        return loss
class Grad_DynamicMarginLoss(torch.nn.Module):
    def __init__(self, device, weight1, weight2, sigma1, sigma2):
        super(Grad_DynamicMarginLoss, self).__init__()
        self.m_device = device
        self.weight1 = weight1
        self.weight2 = weight2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def forward(self, preds, margin):
        with torch.no_grad():


            margin = margin.clone().detach().requires_grad_(True)
            positive_margin_indices = torch.nonzero(margin > 0).squeeze()
            negative_margin_indices = torch.nonzero(margin < 0).squeeze()
            filtered_margin_tensor = torch.zeros(margin.shape[0], dtype=torch.float32, device=self.m_device)
            filtered_margin_tensor[positive_margin_indices] = margin[positive_margin_indices]
            filtered_margin_tensor[negative_margin_indices] = 0.0

            filtered_margin_tensor_neg = torch.zeros(margin.shape[0], dtype=torch.float32, device=self.m_device)
            filtered_margin_tensor_neg[negative_margin_indices] = margin[negative_margin_indices]
            filtered_margin_tensor_neg[positive_margin_indices] = 0.0
            w1 = f(filtered_margin_tensor, self.weight1, self.sigma1)
            w2 = f(filtered_margin_tensor_neg, self.weight2, self.sigma2)
        N_positive_margin = preds[positive_margin_indices]
        filtered_N1 = torch.zeros(margin.shape[0], dtype=preds.dtype, device=self.m_device)
        filtered_N1[positive_margin_indices] = N_positive_margin
        N_negative_margin = preds[negative_margin_indices]
        filtered_N = torch.zeros(margin.shape[0], dtype=preds.dtype, device=self.m_device)
        filtered_N[negative_margin_indices] = N_negative_margin

        loss1 = torch.sum(
            w1 * filtered_N1)
        loss2 = torch.sum(
            w2 * filtered_N)
        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)
        loss = -(loss1 + loss2) / len(margin)
        return loss
def normalize_prop(mx, adj, alpha, n_iter, normFea=False):
    if normFea:
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        # print(r_inv.shape)
        # print(r_inv)
        r_mat_inv = sp.diags(r_inv, 0)
        # r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

    mx = mx.cpu().detach().numpy()
    """Feature propagation via Normalized Laplacian"""
    S = normalize_adj_dense(adj)
    F = alpha * S.dot(mx) + (1 - alpha) * mx
    for _ in range(n_iter):
        F = alpha * S.dot(F) + (1 - alpha) * mx
    return F
class CertifyLoss(torch.nn.Module):
    def __init__(self,  device, a):
        super(CertifyLoss, self).__init__()

        self.m_device = device
        self.a = a

    def forward(self, preds,labels,CertifyK):
        with torch.no_grad():


            weight = 1/(1+torch.exp(self.a*CertifyK))
            weight[weight > 0.5] = 0
        N = preds[torch.arange(preds.shape[0]), labels]
        N = N.to(self.m_device)
        weight = weight.to(self.m_device)
        loss = torch.sum(
            weight*N)

        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)
        loss = -loss / len(labels)
        return loss
class CertifyLossNode(torch.nn.Module):
    def __init__(self,  device, a):
        super(CertifyLossNode, self).__init__()

        self.m_device = device
        self.a = a

    def forward(self, preds,labels,CertifyK):
        with torch.no_grad():


            weight = 1/(1+torch.exp(self.a*CertifyK))
            if weight>0.5:
                weight==0

        N = preds[labels]

        N = N.to(self.m_device)
        weight = weight.to(self.m_device)
        loss = torch.sum(
            weight*N)

        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)
        loss = -loss
        return loss
class CertifyLossCW(torch.nn.Module):
    def __init__(self,  device, a):
        super(CertifyLossCW, self).__init__()

        self.m_device = device
        self.a = a

    def forward(self, logits,labels,CertifyK):
        with torch.no_grad():


            weight = 1/(1+torch.exp(self.a*CertifyK))
            weight[weight > 0.5] = 0
        sorted = logits.argsort(-1)
        best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
        structural_scores = (
                logits[np.arange(logits.size(0)), labels] - logits[np.arange(logits.size(0)), best_non_target_class])

        k = 0
        structural_scores = torch.clamp(structural_scores, min=k)
        weight = weight.to(self.m_device)
        loss = torch.sum(
            weight*structural_scores)

        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)
        loss = -loss / len(labels)
        return loss
def CostCWNode(logits,labels,margin,weight,sigma):


        probs = torch.exp(logits)
        print(probs)
        probs = probs.to('cuda:0')
        # probs_true_label = torch.tensor(0.0)
        probs_true_label = probs[labels]

        probs_true_label = probs_true_label.to('cuda:0')
        probs_new = delete(probs, labels)
        probs_best_second_class = probs_new[probs_new.argmax()]
        print(probs_true_label)
        structuralscores = probs_true_label - probs_best_second_class
        structuralscores = structuralscores.to('cuda:0')
        # print(structuralscores)
        loss = torch.where(structuralscores < 0, torch.tensor(0.0).to('cuda:0'), -structuralscores)

        with torch.no_grad():

            if margin >= 0:
              w1 = f(margin, weight, sigma)
            else:
                w1 =0


        loss = w1*loss
        return loss

        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)


class CertifyLossCWNode(torch.nn.Module):
    def __init__(self,  device, a):
        super(CertifyLossCWNode, self).__init__()

        self.m_device = device
        self.a = a

    def forward(self, logits,labels,CertifyK):
        with torch.no_grad():


            weight = 1/(1+torch.exp(self.a*CertifyK))
            weight[weight > 0.5] = 0
        probs = torch.exp(logits)
        print(probs)
        probs = probs.to('cuda:0')
        # probs_true_label = torch.tensor(0.0)
        probs_true_label = probs[labels]

        probs_true_label = probs_true_label.to('cuda:0')
        probs_new = delete(probs, labels)
        probs_best_second_class = probs_new[probs_new.argmax()]
        print(probs_true_label)
        structuralscores = probs_true_label - probs_best_second_class
        structuralscores = structuralscores.to('cuda:0')
        # print(structuralscores)
        loss = torch.where(structuralscores < 0, torch.tensor(0.0).to('cuda:0'), -structuralscores)

        weight = weight.to(self.m_device)
        loss = torch.sum(
            weight*loss)

        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)

        return loss


class similar_distribution(torch.nn.Module):
    def __init__(self,device, weight1, weight2, sigma1, sigma2):
        super(similar_distribution, self).__init__()
        self.m_device = device
        self.weight1 = weight1
        self.weight2 = weight2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def forward(self, preds, targets, margin):
        # print("==="*10)
        # print(targets.size())

        # print(targets.size())
        # targets = torch.sum(targets, dim=1)

        # targets[:, 0] = 0
        N = preds[torch.arange(preds.shape[0]), targets]
        print(preds.shape)
        positive_margin_indices = torch.nonzero((margin > 0).int()).squeeze()
        negative_margin_indices = torch.nonzero((margin < 0).int()).squeeze()
        N_positive_margin = N[positive_margin_indices]
        filtered_N1 = torch.zeros(margin.shape[0], dtype=N.dtype, device=self.m_device)
        filtered_N1[positive_margin_indices] = N_positive_margin
        N_negative_margin = N[negative_margin_indices]
        filtered_N = torch.zeros(margin.shape[0], dtype=N.dtype, device=self.m_device)
        filtered_N[negative_margin_indices] = N_negative_margin
        with torch.no_grad():
            margin = margin.clone().detach().requires_grad_(True)
            filtered_margin_tensor = torch.zeros(margin.shape[0], dtype=torch.float32, device=self.m_device)
            filtered_margin_tensor[positive_margin_indices] = margin[positive_margin_indices]
            filtered_margin_tensor[negative_margin_indices] = 0.0
            x1 = filtered_margin_tensor.cpu().detach().numpy()
            filtered_margin_tensor_neg = torch.zeros(margin.shape[0], dtype=torch.float32, device=self.m_device)
            filtered_margin_tensor_neg[negative_margin_indices] = margin[negative_margin_indices]
            filtered_margin_tensor_neg[positive_margin_indices] = 0.0
            x2 = filtered_margin_tensor_neg.cpu().detach().numpy()

            w1 = f(filtered_margin_tensor, self.weight1, self.sigma1)
            w2 = f(filtered_margin_tensor_neg, self.weight2, self.sigma2)
        loss1 = torch.sum(
            w1 * filtered_N1)
        loss2 = torch.sum(
            w2 * filtered_N)
        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)
        loss = -(loss1 + loss2) / len(margin)
        # w1 = w1.cpu().detach().numpy()
        # w2 = w2.cpu().detach().numpy()
        # plt.plot(x1, w1, label='y=pos_margin')
        # plt.plot(x2, w2, label='y=neg_margin')
        # #
        # # # 添加标题和标签
        # plt.title("Two Curves")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.legend()  # 显示图例
        # #
        # # 显示图像
        # plt.show()
        # plt.close('all')
        return loss

class similar_distributionode(torch.nn.Module):
    def __init__(self,device, weight1, weight2, sigma1, sigma2):
        super(similar_distributionode, self).__init__()
        self.m_device = device
        self.weight1 = weight1
        self.weight2 = weight2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def forward(self, preds, targets, margin):
        # print("==="*10)
        # print(targets.size())

        # print(targets.size())
        # targets = torch.sum(targets, dim=1)

        # targets[:, 0] = 0
        N = preds[targets]


        with torch.no_grad():
            margin = torch.tensor(margin, device=self.m_device)
            if margin>0:
                w = f(margin, self.weight1, self.sigma1)

            else:
                w = f(margin, self.weight2, self.sigma2)
        loss = -torch.sum(w* N)
        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)

        # w1 = w1.cpu().detach().numpy()
        # w2 = w2.cpu().detach().numpy()
        # plt.plot(x1, w1, label='y=pos_margin')
        # plt.plot(x2, w2, label='y=neg_margin')
        # #
        # # # 添加标题和标签
        # plt.title("Two Curves")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.legend()  # 显示图例
        # #
        # # 显示图像
        # plt.show()
        # plt.close('all')
        return loss

class multi_objective_loss(torch.nn.Module):
    def __init__(self,device, weight1, weight2, sigma1, sigma2):
        super(multi_objective_loss, self).__init__()
        self.m_device = device
        self.weight1 = weight1
        self.weight2 = weight2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def forward(self, preds, targets, margin, score):
        # print("==="*10)
        # print(targets.size())

        # print(targets.size())
        # targets = torch.sum(targets, dim=1)

        # targets[:, 0] = 0
        N = preds[torch.arange(preds.shape[0]), targets]
        positive_margin_indices = torch.nonzero(margin > 0).squeeze()
        negative_margin_indices = torch.nonzero(margin < 0).squeeze()
        N_positive_margin = N[positive_margin_indices]
        filtered_N1 = torch.zeros(margin.shape[0], dtype=N.dtype, device=self.m_device)
        filtered_N1[positive_margin_indices] = N_positive_margin
        N_negative_margin = N[negative_margin_indices]
        filtered_N = torch.zeros(margin.shape[0], dtype=N.dtype, device=self.m_device)
        filtered_N[negative_margin_indices] = N_negative_margin
        # score = score.clone().detach().requires_grad_(True)
        indices = torch.nonzero(score < 0).squeeze()

        with torch.no_grad():
            margin = margin.clone().detach().requires_grad_(True)
            filtered_margin_tensor = torch.zeros(margin.shape[0], dtype=torch.float32, device=self.m_device)
            filtered_margin_tensor[positive_margin_indices] = margin[positive_margin_indices]
            filtered_margin_tensor[negative_margin_indices] = 0.0
            x1 = filtered_margin_tensor.cpu().detach().numpy()
            filtered_margin_tensor_neg = torch.zeros(margin.shape[0], dtype=torch.float32, device=self.m_device)
            filtered_margin_tensor_neg[negative_margin_indices] = margin[negative_margin_indices]
            filtered_margin_tensor_neg[positive_margin_indices] = 0.0
            x2 = filtered_margin_tensor_neg.cpu().detach().numpy()

            w1 = f(filtered_margin_tensor, self.weight1, self.sigma1)
            w2 = f(filtered_margin_tensor_neg, self.weight2, self.sigma2)
        loss1 = torch.sum(
            w1 * filtered_N1)
        loss2 = torch.sum(
            w2 * filtered_N)
        weight_margin = 0.8 # choose an appropriate value for this
        margin_loss = weight_margin * (torch.sum(score[indices]) / len(indices))

        # include margin_loss in the attack_loss

        # print(1 * normal_distribution(filtered_margin_tensor_neg, self.mu, 1) * filtered_N)
        loss = -(loss1 + loss2) / len(margin) + margin_loss
        # loss = torch.sum(score[indices]) / len(indices)
        # for i in range(1):
        #     y1 = (8.5 * normal_distribution(torch.from_numpy(filtered_margin_tensor.cpu().detach().numpy()), self.mu, 2))
        #
        #     y2 = (1 * normal_distribution(torch.from_numpy(filtered_margin_tensor_neg.cpu().detach().numpy()), self.mu, 4))
        # # print(y2)
        # # print(filtered_margin_tensor_neg)
        # # print(filtered_N)
        # w1 = w1.cpu().detach().numpy()
        # w2 = w2.cpu().detach().numpy()
        # plt.plot(x1, w1, label='y=pos_margin')
        # plt.plot(x2, w2, label='y=neg_margin')
        # #
        # # # 添加标题和标签
        # plt.title("Two Curves")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.legend()  # 显示图例
        # #
        # # 显示图像
        # plt.show()
        # plt.close('all')
        return loss


def margin_compute(logits, labels):
    # logits = torch.exp(logits)
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
    margin = (logits[np.arange(logits.size(0)), labels]
              - logits[np.arange(logits.size(0)), best_non_target_class]
              )
    # print(margin)
    # print(margin.shape)
    return margin

# def CWlossNode(logits,labels):
#     sorted = logits.argsort(-1)
#     best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#     structuralscores = (logits[np.arange(logits.size(0)), labels]
#               - logits[np.arange(logits.size(0)), best_non_target_class]
#               )
#     if structuralscores<0:
#         loss =0
#     else:
#         loss = -structuralscores
#
#     return loss

def CWlossNode(logits, labels):
    probs = torch.exp(logits)
    print(probs)
    probs =probs.to('cuda:0')
    # probs_true_label = torch.tensor(0.0)
    probs_true_label = probs[labels]

    probs_true_label = probs_true_label.to('cuda:0')
    probs_new = delete(probs,labels)
    probs_best_second_class = probs_new[probs_new.argmax()]
    print(probs_true_label)
    structuralscores= probs_true_label - probs_best_second_class
    structuralscores = structuralscores.to('cuda:0')
    # print(structuralscores)
    loss = torch.where(structuralscores < 0, torch.tensor(0.0).to('cuda:0'),-structuralscores)
    loss = loss.to('cuda:0')
    return loss
def CWloss(logits,labels):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
    structural_scores = (
            logits[np.arange(logits.size(0)), labels] - logits[np.arange(logits.size(0)), best_non_target_class])
    k = 0
    loss = -torch.clamp(structural_scores, min=k).mean()

    return loss
def labelloss(logits,labels):

    best_non_target_class = 0
    structural_scores = (
            logits[np.arange(logits.size(0)), labels] - logits[np.arange(logits.size(0)), best_non_target_class])
    k = 0
    loss = -torch.clamp(structural_scores, min=k).mean()

    return loss
def weighted_CWloss(logits,labels,margin,weight,sigma,device):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
    structural_scores = (
            logits[np.arange(logits.size(0)), labels] - logits[np.arange(logits.size(0)), best_non_target_class])



    positive_margin_indices = torch.nonzero(margin > 0).squeeze()
    negative_margin_indices = torch.nonzero(margin < 0).squeeze()

    with torch.no_grad():
        marginK = margin.clone().detach().requires_grad_(True)
        filtered_margin_tensor = torch.zeros(marginK.shape[0], dtype=torch.float32, device=device)
        filtered_margin_tensor[positive_margin_indices] = marginK[positive_margin_indices]
        filtered_margin_tensor[negative_margin_indices] = 0.0

        w1 = f(filtered_margin_tensor, weight, sigma)
    k = 0
    marginP = torch.clamp(margin, min=k)

    loss = -torch.sum(w1 * marginP) / len(margin)
    return loss
def structural_scores(logits, labels):
    sorted = logits.argsort(-1)
    best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
    structural_scores = (
            logits[np.arange(logits.size(0)), best_non_target_class] - logits[np.arange(logits.size(0)), labels])
    return structural_scores


def classification_margin_all(output, true_label):
    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def unravel_index(index, array_shape):  # 数组扁平
    rows = torch.div(index, array_shape[1], rounding_mode='trunc')
    cols = index % array_shape[1]
    return rows, cols


def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx
def normalize_adj_dense(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj.cpu().detach().numpy())
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    device = adj.device
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def accuracy(output, labels):
    """Return accuracy of output compared to labels.


    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_list(output, labels):
    """Return accuracy of output compared to labels.


    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    """

    # Determine which degrees are to be considered, i.e. >= d_min.
    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    """ Adopted from https://github.com/danielzuegner/nettack
    """
    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]
    degree_sequence = adjacency_matrix.sum(1)
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)
    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)
    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                            + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min

    new_n = n_old - (old_in_range != 0).sum(1) + (new_in_range != 0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n


def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha = 1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha = 1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll


def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004,
                            undirected=True):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    N = int(modified_adjacency.shape[0])
    # original_degree_sequence = get_degree_squence(original_adjacency)
    # current_degree_sequence = get_degree_squence(modified_adjacency)
    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)

    concat_degree_sequence = torch.cat((current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence,
                                                                                           d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence,
                                                                                           d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.
    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency,
                                                                                               d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)
    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold

    if allowed_edges.is_cuda:
        filtered_edges = node_pairs[allowed_edges.cpu().numpy().astype(bool)]
    else:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(np.bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    if undirected:
        allowed_mask = allowed_mask.clone()
        allowed_mask += allowed_mask + allowed_mask.t()
    return allowed_mask, current_ratio


def is_sparse_tensor(tensor):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


def to_scipy(tensor):
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    """
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    r_mat_inv = sp.diags(r_inv, 0)
    mx = r_mat_inv.dot(mx)
    return mx