#!/usr/bin/env python
# pylint: disable=W0201
import sys
import time
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# loss
from net.loss import Loss

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from processor.processor import Processor
# from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.MSELoss()
        self.loss = Loss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            # self.lr = self.arg.base_lr
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr

    def show_topk(self, k):
        rank = self.result.argsort() # self.result shape (batch_size, num_node)
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_MRR(self):
        # calculate the MRR of the top 1 return-ration stock
        mrr_top = 0.0
        for i in range(self.label.shape[0]) :
            top1_pos_in_gt = 0
            rank_gt = np.argsort(self.label[i])
            rank_pre = np.argsort(self.result[i])
            pre_top1 = set()
            for j in range(1, self.label.shape[1] + 1):
                cur_rank_pre = rank_pre[-1 * j]    # chose the max return-ration stock' id in grouth truth
                cur_rank_gt = rank_gt[-1 * j]
                if len(pre_top1) < 1:
                    pre_top1.add(cur_rank_pre)
                top1_pos_in_gt += 1
                if cur_rank_gt in pre_top1:
                    break
            mrr_top += 1.0 / top1_pos_in_gt
        self.mrrt = mrr_top / self.result.shape[0] 
        # self.io.print_log('\tMRR: {}'.format(mrrt))

    def show_return_ration(self, k):
        # calculate the return-ration of the top k stocks
        self.bt_k = 1.0
        for i in range(self.result.shape[0]):
            rank_pre = np.argsort(self.result[i])

            pre_topk = set()
            for j in range(1, self.result.shape[1] + 1):
                cur_rank = rank_pre[-1 * j]
                if len(pre_topk) < k:
                    pre_topk.add(cur_rank)

            # back testing on top k
            return_ration_topk = 0
            for index in pre_topk:
                return_ration_topk += self.label[i][index]
            return_ration_topk /= k
            with open('IRR_{}.txt'.format(k),'a+') as f:
                f.write(str(round((return_ration_topk),2)))
                f.write('\n')
                f.close()
            self.bt_k += return_ration_topk
            # print(i,return_ration_topk,self.bt_k)
        # self.io.print_log('\tTop{} return ratio: {:.2f}'.format(k, bt_k))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        start_time = time.time()
        for data, closing_price, label in loader:

            # get data
            # print(label.shape,label[:5,:5])
            # exit()
            data = data[:,[0,1,2,4],:,:].float().to(self.dev)
            closing_price = closing_price.float().to(self.dev)
            label = label.float().to(self.dev)
            # forward
            output = self.model(data)
            prediction = torch.div(torch.sub(output, closing_price), closing_price)
            loss = self.loss(prediction, label, 0.1)
            # loss = self.loss(prediction, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_log('Time consumption:{:.4f}s'.format(time.time()-start_time))
        with open('time_consume.txt','a+') as f:
            f.write(str(round((time.time()-start_time),2)))
            f.write('\n')
            f.close()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, closing_price, label in loader:
            
            # get data
            data = data[:,[0,1,2,4],:,:].float().to(self.dev)
            closing_price = closing_price.float().to(self.dev)
            label = label.float().to(self.dev)
            # print(closing_price[:5,:10])
            # inference
            with torch.no_grad():
                output = self.model(data)
                # print(output[:5,:10])
                prediction = torch.div(torch.sub(output, closing_price), closing_price)
            result_frag.append(prediction.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(prediction, label, 0.1)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            # in NYSY dataset, the 89-th day has an waivness influence TOP-n metrixc
            self.result = np.delete(self.result,89, axis=0)
            self.label = np.delete(self.label,89, axis=0)
            mae = 0.0
            for i in range(self.result.shape[0]):
                mae += np.sum(np.absolute(self.result[i]-self.label[i]))/len(self.result[i])
            self.epoch_info['MAE'] = mae / self.result.shape[0]
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()
            self.show_MRR()
            if self.epoch_info['mean_loss'] < self.best_performance['mean_loss']:
                # save the model
                self.io.save_model(self.model, 'best_model{}.pt'.format(self.meta_info['epoch']))
                self.best_performance['mean_loss'] = self.epoch_info['mean_loss']
                self.best_performance['MAE'] = self.epoch_info['MAE']
                self.best_performance['mrr'] = self.mrrt
                # show top-k return ration
                for k in self.arg.show_topk:
                    self.show_return_ration(k)
                    self.best_performance['top'+str(k)] = self.bt_k
        self.io.print_log('\tbest test MAE loss: {}'.format(self.best_performance['MAE']))
        self.io.print_log('\tbest test MRR: {}'.format(self.best_performance['mrr']))
        self.io.print_log('\tTop1 return ratio: {:.2f}'.format(self.best_performance['top1']))
        self.io.print_log('\tTop5 return ratio: {:.2f}'.format(self.best_performance['top5']))
        self.io.print_log('\tTop10 return ratio: {:.2f}'.format(self.best_performance['top10']))    

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Relational Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5, 10], nargs='+', help='which Top K return ration will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
