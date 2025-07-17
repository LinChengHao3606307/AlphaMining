
"""
    上游代理
    act(state)->action_dict:
        把state输入模型，输出{离散动作，连续动作，log(取这套动作的概率)，这套动作的估计收益}
        为后续优化做准备
    store_transition(state, action_dict, reward, done)->buffer_if_full:
        保存历史记录，存不下了上报给train函数，train函数会指挥Agent进入优化
    update()->none:
        拿历史数据优化策略
        具体思想是最小化取曾经产生劣势的动作的概率，最大化曾经产生优势的动作的概率
        做法是把历史数据分批次用以优化策略

"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from __init__ import DefaultValues, Formula

def clamp_positive_ratio(ip: torch.Tensor, min_val: float = 0.2, max_val: float = 0.8) -> torch.Tensor:
    # 确保min和max在0到1之间，并且min <= max
    min_val = max(min_val, 0.0)
    max_val = min(max_val, 1.0)
    if min_val > max_val:
        min_val, max_val = max_val, min_val

    d = ip.size(-1)
    n = (ip > 0).sum(dim=-1)  # 计算最后一个维度中正数的数量
    min_clamp = min_val * d
    max_clamp = max_val * d

    # 确定哪些位置需要调整
    mask_low = n < min_clamp
    mask_high = n > max_clamp
    mask_adjust = mask_low | mask_high

    # 初始化目标k值
    k_target = torch.zeros_like(n, dtype=torch.long)
    # 处理低于min的情况，使用ceil确保至少达到min比例
    k_target[mask_low] = torch.ceil(torch.tensor(min_clamp)).to(torch.long)
    # 处理高于max的情况，使用floor确保不超过max比例
    k_target[mask_high] = torch.floor(torch.tensor(max_clamp)).to(torch.long)
    # 确保k_target在有效范围内
    k_target = torch.clamp(k_target, min=0, max=d-1)

    # 对最后一个维度进行降序排序
    sorted_ip, _ = torch.sort(ip, dim=-1, descending=True)
    # 收集对应的shift值
    shift = torch.gather(sorted_ip, dim=-1, index=k_target.unsqueeze(-1)).squeeze(-1)

    # 应用偏移量，并仅调整需要调整的位置
    adjusted = ip - shift.unsqueeze(-1)
    result = torch.where(mask_adjust.unsqueeze(-1), adjusted, ip)

    return result

def checkNAN(model: nn.Module) -> bool:
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")
            return True
    return False

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)
# using PPO
class HybridPPO:
    def __init__(self, policy, value_net,
                 state_dim = None, state_type = "buf",
                 discrete_dim:int=3, continuous_dim:int=4,
                 lr=1e-3, gamma=1-1/DefaultValues.formula_profile_length, clip_eps=0.5, clip_method="epoch", ent_coef=5e-2, buffer_size=100):
        self.device = DefaultValues.device
        self.policy = policy.to(self.device)
        self.value_net = value_net.to(self.device)
        self.optimizer_actor = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.value_net.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.clip_method = clip_method
        self.ent_coef = ent_coef

        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.continuous_dim = continuous_dim

        if state_type == "tpv":
            state_dim = DefaultValues.tpv_formula_shape
        elif state_type == "buf":
            state_dim = DefaultValues.buf_formula_shape
        else:
            raise NotImplementedError(state_type + " is not implemented")
        self.state_type = state_type
        # 初始化buffer
        self.buffer = {
            'states': np.zeros((buffer_size, state_dim[0], state_dim[1]), dtype=np.float32),
            'discrete_actions': np.zeros((buffer_size, discrete_dim), dtype=np.float32),
            'continuous_actions': np.zeros((buffer_size, continuous_dim), dtype=np.float32) if continuous_dim > 0 else None,
            'log_probs': np.zeros(buffer_size, dtype=np.float32),
            'rewards': np.zeros(buffer_size, dtype=np.float32),
            'dones': np.zeros(buffer_size, dtype=np.bool_),
            'pos': 0  # 当前写入位置
        }

    def save(self,name:str):
        self.policy.save(name)

    def load(self,name:str):
        self.policy.load(name)

    def act(self, state:Formula):
        """处理单个时间步的状态"""
        state = torch.LongTensor(state.buf_formula).unsqueeze(0).to(self.device)

        with torch.no_grad():
            discrete_probs, (mean, std) = self.policy(state)
            # 采样离散动作
            discrete_actions = []
            discrete_log_probs = torch.FloatTensor([0]).to(DefaultValues.device)
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
            total_log_prob = discrete_log_probs #+ continuous_log_prob

            return {
                'discrete': np.array(discrete_actions),
                'continuous': continuous_action,
                'log_prob': total_log_prob.item()
            }


    def store_transition(self, state:Formula, action_dict, reward, done):
        pos = self.buffer['pos']
        if self.state_type == "buf":
            self.buffer['states'][pos] = state.buf_formula
        elif self.state_type == "tpv":
            self.buffer['states'][pos] = state.tpv_formula
        self.buffer['discrete_actions'][pos] = action_dict['discrete']
        self.buffer['continuous_actions'][pos] = action_dict['continuous']
        self.buffer['log_probs'][pos] = action_dict['log_prob']
        self.buffer['rewards'][pos] = reward
        self.buffer['dones'][pos] = done

        # 更新位置指针
        self.buffer['pos'] += 1
        # 检查缓冲区是否已满
        if self.buffer['pos'] >= self.buffer_size:
            self.buffer['pos'] = 0
            return True
        return False

    def update(self, batch_size=32, epochs=5):
        """处理batch和时间步维度"""
        # 准备数据
        states = torch.LongTensor(self.buffer['states']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        discrete_actions = torch.LongTensor(self.buffer['discrete_actions']).to(self.device)
        continuous_actions = torch.LongTensor(self.buffer['continuous_actions']).to(self.device)


        returns, advantages = self._compute_returns(states)


        # 重置缓冲区指针，保留数据结构
        self.buffer['pos'] = 0

        vsl_ratio_records = []

        # 训练循环
        for _ in range(epochs):
            vsl_ratio_record = torch.zeros_like(advantages)
            # 随机打乱所有时间步
            perm = torch.randperm(states.size(0))

            # 小批量更新
            for start in range(0, states.size(0), batch_size):
                end = start + batch_size
                idx = perm[start:end]

                # 获取当前batch数据
                batch_states = states[idx]
                batch_discrete_actions = discrete_actions[idx]
                batch_continuous_actions = continuous_actions[idx] if continuous_actions is not None else None
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                # 评估当前策略
                discrete_probs, (mean, std) = self.policy(batch_states)
                predicted_values = self.value_net(batch_states)

                # 计算离散动作的log概率和熵
                discrete_log_probs = torch.zeros([idx.shape[0]]).to(DefaultValues.device)
                discrete_entropies = []

                for i, probs in enumerate(discrete_probs):
                    action = batch_discrete_actions[:, i]
                    dist = Categorical(probs)
                    discrete_log_probs += dist.log_prob(action)
                    #print(f"probs: {probs}, action: {action}, log_prob: {discrete_log_probs}")
                    discrete_entropies.append(dist.entropy())

                # 计算连续动作的log概率和熵
                continuous_log_prob = 0
                continuous_entropy = 0
                if mean is not None:
                    dist = Normal(mean, std)
                    continuous_log_prob = dist.log_prob(batch_continuous_actions).sum(dim=-1)
                    continuous_entropy = dist.entropy().sum(dim=-1)
                # 合并所有动作的log概率和熵
                """print("2 "*20)
                print(discrete_log_probs)
                print(continuous_log_prob)"""
                total_log_prob = discrete_log_probs #+ continuous_log_prob
                total_entropy = sum(discrete_entropies) #+ continuous_entropy

                # 计算比率
                #print(total_log_prob)
                #print(batch_old_log_probs)
                ratio = (total_log_prob - batch_old_log_probs).exp()
                # PPO损失
                vsl_ratio_record[idx] = ratio.clone()
                surr1 = ratio * batch_advantages
                if self.clip_method == "epoch":
                    up_clip = (1 + self.clip_eps)**(1 / epochs)
                    lo_clip = (1 - self.clip_eps)**(1 / epochs)
                else:
                    up_clip = (1 + self.clip_eps)
                    lo_clip = (1 - self.clip_eps)
                surr2 = torch.clamp(ratio, lo_clip, up_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = 0.5 * (predicted_values.squeeze() - batch_returns).pow(2).mean()
                # 熵正则化
                entropy_loss = -self.ent_coef * total_entropy.mean()
                # 总损失
                loss = policy_loss + entropy_loss
                # 梯度更新
                self.optimizer_actor.zero_grad()

                loss.backward()
                #nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                value_loss.backward()
                self.optimizer_critic.step()

            vsl_ratio_records.append(vsl_ratio_record)

        f_str = " || "
        f_str += ".   " * DefaultValues.fn_profile_length
        for i, s in enumerate(states):
            print(" ====RE====  ====AD==== ",end="")
            print(" ______ " * epochs, end="")

            print(f_str * DefaultValues.formula_profile_length)
            if returns[i]>0:
                print(f"{returns[i]:.5f}\t\t",end="")
            else:
                print(f"{returns[i]:.5f}\t",end="")
            if advantages[i]>0:
                print(f"{advantages[i]:.5f}\t\t",end="")
            else:
                print(f"{advantages[i]:.5f}\t",end="")
            for e in range(epochs):
                print(f"{vsl_ratio_records[e][i]:.4f}\t", end="")
            for fn in s:
                print(" || ", end="")
                for num in fn:
                    print(num.item(),end="\t")
            print()





    def _compute_returns(self, states, use_value_net = 0.6, use_mean = True, use_std = True, use_ratio_c = False):
        """计算GAE和returns"""
        rewards = torch.FloatTensor(self.buffer['rewards']).to(self.device)
        dones = self.buffer['dones']
        """
        for i in range(states.shape[0]):
            print(states[i].cpu().numpy(),end=" > ")
            print(dones[i])
        """
        # 计算GAE
        advantages = torch.zeros_like(rewards).to(self.device).float()
        last_gae = 0
        for t in reversed(range(rewards.shape[0])):
            advantages[t] = last_gae = rewards[t] + self.gamma * (1 - dones[t]) * last_gae

        # 计算returns并标准化advantages
        returns = advantages.clone()
        values = self.value_net(states).squeeze().detach()
        advantages -= values*use_value_net
        if use_mean:
            x = advantages.mean()
            advantages -= x
            #const_m = 4
            #advantages -= const_m*x / (const_m + torch.sqrt(x.abs()))
        if use_ratio_c:
            advantages = clamp_positive_ratio(advantages,min_val=0.3,max_val=0.7)
        if use_std:
            advantages /= (advantages.std() + 1e-8)
        return returns, advantages


