"""
改进版的PPO Agent实现
支持更灵活的配置和更好的训练稳定性
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from __init__ import DefaultValues, Formula


class ImprovedPPO:
    def __init__(self, 
                 policy=None, 
                 value_net=None,
                 state_type="buf",
                 lr=1e-3, 
                 gamma=0.8, 
                 clip_eps=0.2, 
                 clip_method="epoch", 
                 ent_coef=0.01, 
                 buffer_size=1000,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 target_kl=0.01):
        """
        改进版PPO Agent
        
        Args:
            policy: 策略网络
            value_net: 价值网络
            state_type: 状态类型 ("buf", "tpv", "pse")
            lr: 学习率
            gamma: 折扣因子
            clip_eps: PPO裁剪参数
            clip_method: 裁剪方法 ("epoch", "step")
            ent_coef: 熵系数
            buffer_size: 缓冲区大小
            value_coef: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
            target_kl: 目标KL散度
        """
        self.device = DefaultValues.device
        self.policy = None
        self.value_net = None
        self.state_type = state_type
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.clip_method = clip_method
        self.ent_coef = ent_coef
        self.buffer_size = buffer_size
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # 使用Formula类来获取正确的状态维度
        formula = Formula()
        state_dim = formula.get(state_type, want_size=True)
        
        self.state_dim = state_dim
        
        # 初始化缓冲区
        self.buffer = {
            'states': np.zeros((buffer_size, state_dim[0], state_dim[1]), dtype=np.float32),
            'discrete_actions': np.zeros((buffer_size, 3), dtype=np.float32),
            'continuous_actions': np.zeros((buffer_size, 4), dtype=np.float32),
            'log_probs': np.zeros(buffer_size, dtype=np.float32),
            'rewards': np.zeros(buffer_size, dtype=np.float32),
            'dones': np.zeros(buffer_size, dtype=np.bool_),
            'values': np.zeros(buffer_size, dtype=np.float32),
            'pos': 0
        }
        
        # 优化器将在设置网络后初始化
        self.optimizer_actor = None
        self.optimizer_critic = None
        
        # 如果提供了网络，立即设置
        if policy is not None:
            self.set_policy(policy)
        if value_net is not None:
            self.set_value_net(value_net)
    
    def set_policy(self, policy):
        """设置策略网络"""
        self.policy = policy.to(self.device)
        if self.optimizer_actor is None:
            self.optimizer_actor = optim.Adam(self.policy.parameters(), lr=self.lr)
    
    def set_value_net(self, value_net):
        """设置价值网络"""
        self.value_net = value_net.to(self.device)
        if self.optimizer_critic is None:
            self.optimizer_critic = optim.Adam(self.value_net.parameters(), lr=self.lr)
    
    def act(self, state):
        """选择动作"""
        state_tensor = torch.LongTensor(state.get(self.state_type)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            discrete_probs, (mean, std) = self.policy(state_tensor)
            
            # 采样离散动作
            discrete_actions = []
            discrete_log_probs = torch.FloatTensor([0]).to(self.device)
            
            for i, probs in enumerate(discrete_probs):
                dist = Categorical(probs)
                action = dist.sample()
                discrete_actions.append(action.item())
                discrete_log_probs += dist.log_prob(action)
            
            # 采样连续动作
            continuous_action = None
            continuous_log_prob = 0
            if mean is not None:
                dist = Normal(mean, std)
                action = torch.round(dist.sample()).relu()
                continuous_log_prob = dist.log_prob(action).sum(dim=-1)
                continuous_action = action.squeeze().clone().cpu().numpy()
            
            # 计算总log概率
            total_log_prob = discrete_log_probs
            
            # 计算价值
            value = self.value_net(state_tensor).item()
            
            return {
                'discrete': np.array(discrete_actions),
                'continuous': continuous_action,
                'log_prob': total_log_prob.item(),
                'value': value
            }
    
    def store_transition(self, state, action_dict, reward, done):
        """存储转移"""
        pos = self.buffer['pos']
        self.buffer['states'][pos] = state.get(self.state_type)
        
        self.buffer['discrete_actions'][pos] = action_dict['discrete']
        self.buffer['continuous_actions'][pos] = action_dict['continuous']
        self.buffer['log_probs'][pos] = action_dict['log_prob']
        self.buffer['rewards'][pos] = reward
        self.buffer['dones'][pos] = done
        self.buffer['values'][pos] = action_dict.get('value', 0)
        
        # 更新位置指针
        self.buffer['pos'] += 1
        if self.buffer['pos'] >= self.buffer_size:
            self.buffer['pos'] = 0
            return True
        return False
    
    def update(self, batch_size=32, epochs=5):
        """更新策略和价值网络"""
        
        # 准备数据
        states = torch.LongTensor(self.buffer['states'][:self.buffer['pos']]).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs'][:self.buffer['pos']]).to(self.device)
        discrete_actions = torch.LongTensor(self.buffer['discrete_actions'][:self.buffer['pos']]).to(self.device)
        continuous_actions = torch.LongTensor(self.buffer['continuous_actions'][:self.buffer['pos']]).to(self.device)
        rewards = torch.FloatTensor(self.buffer['rewards'][:self.buffer['pos']]).to(self.device)
        dones = torch.BoolTensor(self.buffer['dones'][:self.buffer['pos']]).to(self.device)
        old_values = torch.FloatTensor(self.buffer['values'][:self.buffer['pos']]).to(self.device)
        
        # 计算回报和优势
        returns, advantages = self._compute_returns_and_advantages(rewards, dones, old_values)
        
        # 重置缓冲区
        self.buffer['pos'] = 0
        
        # 训练循环
        for epoch in range(epochs):
            # 随机打乱数据
            perm = torch.randperm(states.size(0))
            
            for start in range(0, states.size(0), batch_size):
                print(">>>>>>")
                end = start + batch_size
                idx = perm[start:end]
                
                # 获取batch数据
                batch_states = states[idx]
                batch_discrete_actions = discrete_actions[idx]
                batch_continuous_actions = continuous_actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # 评估当前策略
                discrete_probs, (mean, std) = self.policy(batch_states)
                predicted_values = self.value_net(batch_states).squeeze()
                
                # 计算新的log概率
                discrete_log_probs = torch.zeros([idx.shape[0]]).to(self.device)
                discrete_entropies = []
                
                for i, probs in enumerate(discrete_probs):
                    action = batch_discrete_actions[:, i]
                    dist = Categorical(probs)
                    discrete_log_probs += dist.log_prob(action)
                    discrete_entropies.append(dist.entropy())
                
                # 计算连续动作的log概率
                continuous_log_prob = 0
                continuous_entropy = 0
                if mean is not None:
                    dist = Normal(mean, std)
                    continuous_log_prob = dist.log_prob(batch_continuous_actions).sum(dim=-1)
                    continuous_entropy = dist.entropy().sum(dim=-1)
                
                total_log_prob = discrete_log_probs
                total_entropy = torch.stack(discrete_entropies).sum(dim=0)
                
                # 计算比率
                ratio = (total_log_prob - batch_old_log_probs).exp()
                
                # PPO损失
                surr1 = ratio * batch_advantages
                if self.clip_method == "epoch":
                    up_clip = (1 + self.clip_eps) ** (1 / epochs)
                    lo_clip = (1 - self.clip_eps) ** (1 / epochs)
                else:
                    up_clip = (1 + self.clip_eps)
                    lo_clip = (1 - self.clip_eps)
                
                surr2 = torch.clamp(ratio, lo_clip, up_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = 0.5 * (predicted_values - batch_returns).pow(2).mean()
                
                # 熵正则化
                entropy_loss = -self.ent_coef * total_entropy.mean()
                
                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
                
                # 更新策略网络
                self.optimizer_actor.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()
                
                # 更新价值网络
                self.optimizer_critic.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()
    
    def _compute_returns_and_advantages(self, rewards, dones, values):
        """计算回报和优势函数"""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # 计算回报
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # 计算优势函数（使用GAE）
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae
            advantages[t] = gae
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def save(self, name):
        """保存模型"""
        if self.policy:
            self.policy.save(name)
    
    def load(self, name):
        """加载模型"""
        if self.policy:
            self.policy.load(name) 