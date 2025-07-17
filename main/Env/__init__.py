
"""
    下游环境
    reset()->none:
        重置环境，清空dataset里已经算好的公式以及有效公式的标记
    step(action)->next_state, reward, done, vali_loss:
        根据模型选择的函数以及参数:
            如果模型选择结束或公式已经达到最大长度->把公式拿去结算，奖励模型并返回空公式
                结算步骤
                    检查DataSet有无空位
                    选择空位或根据最差公式的代号，替换DataSet里的旧公式，调用DataSet自我更新的特化设计，更新DataSet
                    用新DataSet训练下游模型
                    评选最差公式，记住代号
                    返回训练结果
            如果模型还在添加函数->先不奖励，把函数接在之前的公式后面返回给模型继续
"""

import time

from main.uti.train import Train_data_holder
import math

import numpy as np
import torch
import torch.nn as nn
from .DataSet import DataSet, SubSet, UnionSet
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from __init__ import DefaultValues, Formula


class Env:
    def __init__(self, original_data:list[pd.DataFrame],
                 split_idx:tuple[int,int],
                 lower_model_class,
                 criterion:nn.Module,
                 num_formulas:int = DefaultValues.total_amount_of_formulas,
                 formula_profile_length: int = DefaultValues.formula_profile_length,
                 lower_train_epoch:int=1, lower_train_batch:int=256, lower_train_lr:int=0.001,
                 bound:tuple[float,float] = (-1,4),
                 sign_division:float=0.005):
        self.state:Formula = Formula()
        self.num_stock = len(original_data)
        self.data_sets = [DataSet(d, num_formulas) for d in original_data]
        same_len_sets = []
        for d in self.data_sets:
            if len(self.data_sets[0]) == len(d):
                same_len_sets.append(d)
        assert len(same_len_sets)/self.num_stock > 0.7
        self.num_stock = len(same_len_sets)
        self.data_sets = same_len_sets

        self.num_formulas = num_formulas + 1
        self.valid_formulas: np.ndarray = np.zeros(self.num_formulas)
        self.formula_profile_length = formula_profile_length
        self.lower_model_class = lower_model_class

        self.criterion = criterion.to(DefaultValues.device)
        self.lower_train_epoch = lower_train_epoch
        self.lower_train_batch = lower_train_batch
        self.lower_train_lr = lower_train_lr
        self.split_idx = split_idx
        self.ip_shape = self.data_sets[0][0][1].shape
        self.tdh_r = Train_data_holder(["reward", "zero"],title="reward - min_reward")
        self.tdh_l = Train_data_holder(["test_loss", "vali_loss"])
        assert len(self.ip_shape) == 1
        self.id_ip = torch.eye(self.ip_shape[0], self.ip_shape[0]).to(DefaultValues.device)
        self.formulas_idx_to_be_replace = -1
        self.bound = bound
        assert self.bound[1] > 0 > self.bound[0]
        assert  sign_division > 0
        self.upper_bound_result = (self.bound[1] - self.bound[0])**(-0.5)
        b = (-self.bound[0])**(-0.5) - self.upper_bound_result
        self.const = b/sign_division

    def reset(self):
        self.state = Formula()
        self.valid_formulas = [0 for _ in self.valid_formulas]
        for i in range(self.num_stock):
            self.data_sets[i].reset()
        return self.state

    def step(self,action):
        vali_loss = None
        reward = 0
        done = 0
        if not action[0] == -1:
            self.state.append_std_fn(action)
        if Formula.is_dummy(action, len(self.state)):
            return self.state, self.bound[0], done, vali_loss
        if action[0] == -1 or len(self.state) == self.formula_profile_length:
            done = 1
            if len(self.state) == 0:
                return self.state, self.bound[0], done, vali_loss
            i=0
            while self.valid_formulas[i] == 1:
                i += 1
                if i == self.num_formulas:
                    i = self.formulas_idx_to_be_replace
                    break
            for set_i in range(self.num_stock):
                self.data_sets[set_i].replace_formula(self.state.std_formula,0)
            self.valid_formulas[i] = 1
            tt_loss, vali_loss, scores = self._train_lower()
            reward = (self.upper_bound_result + self.const * tt_loss)**(-2) + self.bound[0]
            self.tdh_r.add_record_to_current_epoch("reward", reward-self.bound[0])
            self.tdh_r.add_record_to_current_epoch("zero",-self.bound[0])
            self.tdh_r.log_past_and_current()
            self.tdh_r.plot_graph()
            self.tdh_l.add_record_to_current_epoch("test_loss", tt_loss)
            self.tdh_l.add_record_to_current_epoch( "vali_loss", vali_loss)
            self.tdh_l.log_past_and_current()
            self.tdh_l.plot_graph()
            formulas_score = scores[-self.num_formulas:] * self.valid_formulas

            self.formulas_idx_to_be_replace = np.argmin(formulas_score)

            self.state = Formula()



        return self.state, reward, done, vali_loss

    def _train_lower(self,vali_on:bool=True):
        '''model = self.lower_model_class(
            DefaultValues.total_amount_of_factors + DefaultValues.total_amount_of_formulas + 1
        ).to(DefaultValues.device)'''
        model = self.lower_model_class(
            DefaultValues.total_amount_of_formulas + 1
        ).to(DefaultValues.device)
        t_lose, vali_loss = 0, 0
        train_set = UnionSet(
            [SubSet(s, (
            0, self.split_idx[0]
            )) for s in self.data_sets]
        )
        test_set = UnionSet(
            [SubSet(s, (
            self.split_idx[0], self.split_idx[1]
            )) for s in self.data_sets]
        )
        if vali_on:
            vali_set = UnionSet(
                [SubSet(s, (
                self.split_idx[1], len(self.data_sets[0])
                )) for s in self.data_sets]
            )
        optimizer = optim.Adam(model.parameters(), lr=self.lower_train_lr, weight_decay=0.001)
        length = len(train_set) + len(test_set)
        for epoch in range(self.lower_train_epoch):
            train_dataloader = DataLoader(train_set, batch_size=self.lower_train_batch, shuffle=True)
            model.train()
            for tgt_op, ip in train_dataloader:

                optimizer.zero_grad()
                model_op: torch.Tensor = model(ip)
                tgt_op.requires_grad = True
                loss = self.criterion(model_op, tgt_op)
                t_lose += loss / length
                loss.backward()
                optimizer.step()

            test_dataloader = DataLoader(test_set, batch_size=self.lower_train_batch, shuffle=True)
            model.eval()
            for tgt_op, ip in test_dataloader:
                with torch.no_grad():
                    model_op = model(ip)
                    loss = self.criterion(model_op, tgt_op)
                    t_lose += loss / length
            if vali_on:
                vali_dataloader = DataLoader(vali_set, batch_size=self.lower_train_batch, shuffle=True)
                model.eval()
                length = len(vali_set)
                for tgt_op, ip in vali_dataloader:
                    with torch.no_grad():
                        model_op = model(ip)
                        loss = self.criterion(model_op, tgt_op)
                        vali_loss += loss / length
        scores:torch.Tensor = model(self.id_ip)-model(self.id_ip*0)
        return (
            t_lose.cpu().item()/self.lower_train_epoch,
            vali_loss.cpu().item()/self.lower_train_epoch,
            scores.squeeze().detach().cpu().numpy()
        )
