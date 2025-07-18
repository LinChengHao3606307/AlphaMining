"""
改进版的环境实现
支持更灵活的配置和更好的训练稳定性
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from __init__ import DefaultValues, Formula
from main.Env import Env
from main.Env.models.Linear_ac import Linear_ac
from main.uti.train import Train_data_holder


class ImprovedEnv:
    def __init__(self, 
                 original_data=None,
                 split_idx=None,
                 lower_model_class=None,
                 criterion=None,
                 num_formulas=1,
                 formula_profile_length=5,
                 lower_train_epoch=1,
                 lower_train_batch=256,
                 lower_train_lr=0.001,
                 bound=(-1, 4),
                 sign_division=0.005,
                 use_early_stopping=True,
                 patience=5,
                 min_delta=1e-4):
        """
        改进版环境
        
        Args:
            original_data: 原始数据
            split_idx: 数据分割索引
            lower_model_class: 下游模型类
            criterion: 损失函数
            num_formulas: 公式数量
            formula_profile_length: 公式长度
            lower_train_epoch: 下游训练轮数
            lower_train_batch: 下游训练批次大小
            lower_train_lr: 下游学习率
            bound: 奖励边界
            sign_division: 符号分割
            use_early_stopping: 是否使用早停
            patience: 早停耐心值
            min_delta: 最小改善阈值
        """
        self.original_data = original_data
        self.split_idx = split_idx
        self.lower_model_class = lower_model_class or Linear_ac
        self.criterion = criterion or nn.MSELoss()
        self.num_formulas = num_formulas
        self.formula_profile_length = formula_profile_length
        self.lower_train_epoch = lower_train_epoch
        self.lower_train_batch = lower_train_batch
        self.lower_train_lr = lower_train_lr
        self.bound = bound
        self.sign_division = sign_division
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.min_delta = min_delta
        
        # 状态和数据集
        self.state = None
        self.data_sets = None
        self.valid_formulas = None
        self.formulas_idx_to_be_replace = -1
        
        # 训练记录器
        self.tdh_r = None
        self.tdh_l = None
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 如果提供了数据，立即初始化
        if original_data is not None and split_idx is not None:
            self._initialize()
    
    def setup(self, original_data, split_idx, target_formula=None, **kwargs):
        """设置环境参数"""
        self.original_data = original_data
        self.split_idx = split_idx
        
        # 更新其他参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self._initialize()
    
    def _initialize(self):
        """初始化环境"""
        from main.Env.DataSet import DataSet
        
        self.state = Formula()
        self.num_stock = len(self.original_data)
        self.data_sets = [DataSet(d, self.num_formulas) for d in self.original_data]
        
        # 确保数据集长度一致
        same_len_sets = []
        for d in self.data_sets:
            if len(self.data_sets[0]) == len(d):
                same_len_sets.append(d)
        assert len(same_len_sets) / self.num_stock > 0.7
        self.num_stock = len(same_len_sets)
        self.data_sets = same_len_sets
        
        self.num_formulas = self.num_formulas + 1
        self.valid_formulas = np.zeros(self.num_formulas)
        
        # 初始化训练记录器
        self.tdh_r = Train_data_holder(["reward", "zero"], title="reward - min_reward")
        self.tdh_l = Train_data_holder(["test_loss", "vali_loss"])
        
        # 计算奖励常数
        assert self.bound[1] > 0 > self.bound[0]
        assert self.sign_division > 0
        self.upper_bound_result = (self.bound[1] - self.bound[0]) ** (-0.5)
        b = (-self.bound[0]) ** (-0.5) - self.upper_bound_result
        self.const = b / self.sign_division
    
    def reset(self):
        """重置环境"""
        self.state = Formula()
        self.valid_formulas = np.zeros(self.num_formulas)
        for i in range(self.num_stock):
            self.data_sets[i].reset()
        return self.state
    
    def step(self, action):
        """执行一步"""
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
            
            # 找到可用的公式槽位
            i = 0
            while self.valid_formulas[i] == 1:
                i += 1
                if i == self.num_formulas:
                    i = self.formulas_idx_to_be_replace
                    break
            
            # 替换公式
            for set_i in range(self.num_stock):
                self.data_sets[set_i].replace_formula(self.state.std_formula, 0)
            self.valid_formulas[i] = 1
            
            # 训练下游模型
            tt_loss, vali_loss, scores = self._train_lower_improved()
            
            # 计算奖励
            reward = (self.upper_bound_result + self.const * tt_loss) ** (-2) + self.bound[0]
            
            # 记录训练结果
            self.tdh_r.add_record_to_current_epoch("reward", reward - self.bound[0])
            self.tdh_r.add_record_to_current_epoch("zero", -self.bound[0])
            self.tdh_r.log_past_and_current()
            self.tdh_r.plot_graph()
            
            self.tdh_l.add_record_to_current_epoch("test_loss", tt_loss)
            self.tdh_l.add_record_to_current_epoch("vali_loss", vali_loss)
            self.tdh_l.log_past_and_current()
            self.tdh_l.plot_graph()
            
            # 更新最差公式索引
            formulas_score = scores[-self.num_formulas:] * self.valid_formulas
            self.formulas_idx_to_be_replace = np.argmin(formulas_score)
            
            # 重置状态
            self.state = Formula()
        
        return self.state, reward, done, vali_loss
    
    def _train_lower_improved(self, vali_on=True):
        """改进的下游模型训练"""
        from main.Env.DataSet import UnionSet, SubSet
        
        # 创建模型
        model = self.lower_model_class(
            DefaultValues.total_amount_of_formulas + 1
        ).to(DefaultValues.device)
        
        # 创建数据集
        train_set = UnionSet([
            SubSet(s, (0, self.split_idx[0])) for s in self.data_sets
        ])
        test_set = UnionSet([
            SubSet(s, (self.split_idx[0], self.split_idx[1])) for s in self.data_sets
        ])
        
        if vali_on:
            vali_set = UnionSet([
                SubSet(s, (self.split_idx[1], len(self.data_sets[0]))) for s in self.data_sets
            ])
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=self.lower_train_lr, weight_decay=0.001)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.lower_train_epoch):
            # 训练阶段
            model.train()
            train_loss = 0
            train_dataloader = DataLoader(train_set, batch_size=self.lower_train_batch, shuffle=True)
            
            for tgt_op, ip in train_dataloader:
                optimizer.zero_grad()
                model_op = model(ip)
                loss = self.criterion(model_op, tgt_op)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 测试阶段
            model.eval()
            test_loss = 0
            test_dataloader = DataLoader(test_set, batch_size=self.lower_train_batch, shuffle=False)
            
            with torch.no_grad():
                for tgt_op, ip in test_dataloader:
                    model_op = model(ip)
                    loss = self.criterion(model_op, tgt_op)
                    test_loss += loss.item()
            
            # 验证阶段
            vali_loss = 0
            if vali_on:
                vali_dataloader = DataLoader(vali_set, batch_size=self.lower_train_batch, shuffle=False)
                model.eval()
                
                with torch.no_grad():
                    for tgt_op, ip in vali_dataloader:
                        model_op = model(ip)
                        loss = self.criterion(model_op, tgt_op)
                        vali_loss += loss.item()
                
                vali_loss /= len(vali_set)
                
                # 早停检查
                if self.use_early_stopping:
                    if vali_loss < best_val_loss - self.min_delta:
                        best_val_loss = vali_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                # 学习率调度
                scheduler.step(vali_loss)
            
            # 计算平均损失
            train_loss /= len(train_set)
            test_loss /= len(test_set)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={train_loss:.4f}, Test={test_loss:.4f}, Vali={vali_loss:.4f}")
        
        # 计算最终分数
        model.eval()
        id_ip = torch.eye(
            DefaultValues.total_amount_of_formulas + 1, 
            DefaultValues.total_amount_of_formulas + 1
        ).to(DefaultValues.device)
        
        with torch.no_grad():
            scores = model(id_ip) - model(id_ip * 0)
        
        return (
            test_loss,
            vali_loss if vali_on else 0,
            scores.squeeze().detach().cpu().numpy()
        )
    
    def start_new_epoch(self):
        """开始新的训练轮次"""
        self.tdh_r.start_new_epoch_record()
        self.tdh_l.start_new_epoch_record()
    
    def get_last_epoch_avg_loss(self, category="reward"):
        """获取上一轮的平均损失"""
        return self.tdh_r.get_last_epoch_avg_loss(category) 