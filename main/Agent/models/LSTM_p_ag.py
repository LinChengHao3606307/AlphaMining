
"""
    上游策略生成神经网络
    可优化组件
        self.factors_embedding_l:
            包含网络对factor的记忆，如果函数是访问factor的，此长期记忆嵌入层会根据factor编号调用数据输入到lstm的长期记忆中
        self.factors_embedding_s:
            包含网络对factor的记忆，如果函数是访问factor的，此短期记忆嵌入层会根据factor编号调用数据输入到lstm的短期记忆中
        self.fn_embedding
            包含网络对function的记忆，此嵌入层会根据function编号调用数据输入到lstm的input中
        self.fn_profile_fuser
            网络用来根据function参数调整function对lstm的输入的网络
        self.sm_fuser
            公式树交汇处短期记忆融合网络
        self.lm_fuser
            公式树交汇处长期记忆融合网络
        self.lstm_cell
            lstm主体
        self.fn_types_output = nn.Linear(lstm_hidden_size, total_types_of_fn)
            输出模型想添加的函数的类型
        self.fn_para_mean_output = nn.Linear(lstm_hidden_size, DefaultValues.fn_profile_length-1)
            输出模型想添加的函数的参数值
        self.fn_para_std_output = nn.Linear(lstm_hidden_size, DefaultValues.fn_profile_length-1)
            输出模型想添加的函数的参数值的震荡，表明模型的探索意愿
        self.value_output = nn.Linear(lstm_hidden_size, 1)
            输出模型对此次行为的价值的估计
    主要方法
        _get(self, all_formulas, idx, t_shift, t_span)->(sm,lm):
            递归计算
            模型对:
                在已有的公式树上，于时间段[t_shift, t_shift+t_span]调用all_formulas[idx]的行为
            的印象
            返回lstm的(短期记忆，长期记忆)元组
        forward(input_state)->action:
            在公式树根部调用_get()
            根据结果输出action

"""
import os.path

import torch
import torch.nn as nn
from numpy.ma.core import shape

from __init__ import DefaultValues, Formula, get_model_state_dict_path
import torch.nn.functional as F

assert DefaultValues.fn_profile_length == 7


class LSTM_p_ag(nn.Module):
    def __init__(
            self,
            total_amount_of_factors: int = DefaultValues.total_amount_of_factors,
            total_types_of_fn: int = DefaultValues.total_types_of_fn,
            formula_profile_length: int = DefaultValues.formula_profile_length,
            lstm_hidden_size: int = 128,
            lstm_num_layers:int = 3,
            save_path:str = os.path.join(DefaultValues.model_state_dict_path,"LSTM_p_ag")

    ):
        self.save_path = save_path
        super().__init__()
        self.total_amount_of_factors = total_amount_of_factors
        # discrete_probs, (mean, std), value = self.policy(state)
        self.para_scale = (formula_profile_length+total_amount_of_factors)
        #all EBD
        self.factors_embedding_l = nn.Sequential(
            nn.Hardtanh(min_val=0-0.5,max_val=total_amount_of_factors-0.5),
            nn.Embedding(total_amount_of_factors, lstm_hidden_size*lstm_num_layers)
        )
        self.factors_embedding_s = nn.Sequential(
            nn.Hardtanh(min_val=0-0.5,max_val=total_amount_of_factors-0.5),
            nn.Embedding(total_amount_of_factors, lstm_hidden_size*lstm_num_layers)
        )
        self.fn_embedding = nn.Sequential(
            nn.Hardtanh(min_val=0 - 0.5, max_val=total_types_of_fn - 0.5),
            nn.Embedding(total_types_of_fn, lstm_hidden_size)
        )

        self.formula_profile_length = formula_profile_length
        self.fn_profile_fuser = nn.Sequential(
            nn.Linear(lstm_hidden_size+2, lstm_hidden_size*2),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size*2, lstm_hidden_size)
        )
        """
        self.sm_fuser = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, lstm_hidden_size)
        )
        self.lm_fuser = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, lstm_hidden_size)
        )
        """

        """
        self.lstm_cell = nn.LSTMCell(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
        )
        """
        self.lstm_cell = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=0.2
        )

        self.bilstm_summerizer = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=0.2,
            bidirectional=True
        )
        self.lstm_num_layers = lstm_num_layers
        # action output
        self.fn_types_output = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, lstm_hidden_size*2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size*2, total_types_of_fn)
        )
        self.ip_idx_dim = DefaultValues.formula_profile_length + DefaultValues.total_amount_of_factors


        self.ip1_idx_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, 2 * self.ip_idx_dim),
            nn.ReLU(),
            nn.Linear(2 * self.ip_idx_dim, self.ip_idx_dim)
        )
        self.ip2_idx_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, 2 * self.ip_idx_dim),
            nn.ReLU(),
            nn.Linear(2 * self.ip_idx_dim, self.ip_idx_dim)
        )



        self.fn_para_mean_output = nn.Linear(lstm_hidden_size*2, DefaultValues.fn_profile_length-3)
        self.fn_para_std_output = nn.Linear(lstm_hidden_size*2, DefaultValues.fn_profile_length-3)

        self.lstm_hidden_size = lstm_hidden_size
        self.drop = nn.Dropout(0.1)

        """hsb = torch.zeros(
            [self.total_amount_of_factors + DefaultValues.formula_profile_length, self.lstm_hidden_size]
        ).to(DefaultValues.device)
        self.register_buffer('hidden_states_buffer', hsb )"""
        self.hidden_states_buffer = None



        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize embeddings with uniform distribution
        for embedding in [self.factors_embedding_l, self.factors_embedding_s, self.fn_embedding]:
            nn.init.uniform_(embedding[1].weight, -0.1, 0.1)  # embedding[1] because it's wrapped in Sequential

        # Initialize all LSTM weights
        for name, param in self.lstm_cell.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 (common practice for LSTMs)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        for name, param in self.bilstm_summerizer.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 (common practice for LSTMs)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

        # Initialize output layers with smaller weights
        for layer in self.fn_types_output:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        nn.init.uniform_(self.fn_para_mean_output.weight, -0.01, 0.01)
        nn.init.uniform_(self.fn_para_std_output.weight, -0.01, 0.01)

    def _get(self, all_formulas, idx, t_shift, t_span): #num_layers, b, hidden_size
        if idx.detach().cpu().item() < 0:

            #print(f"EBD calling factors_embedding with input = {-idx.detach().cpu().item()-1}")
            return (
                self.factors_embedding_s(-idx-1).view(self.lstm_num_layers,1,self.lstm_hidden_size),
                self.factors_embedding_l(-idx-1).view(self.lstm_num_layers,1,self.lstm_hidden_size)
            )
        type_idx, id1, t_shift1, t_span1, id2, t_shift2, t_span2 = all_formulas[idx.item()].split(1)

        if type_idx.item() == DefaultValues.getter_fn_idx:
            return self._get(all_formulas, id1, t_shift1, t_span1)
        #print(f"EBD calling fn_embedding with input = {type_idx.detach().cpu().item()}")
        current_ip = torch.cat([self.fn_embedding(type_idx), t_shift.unsqueeze(0), t_span.unsqueeze(0)], dim=-1)
        current_ip:torch.Tensor = self.fn_profile_fuser(current_ip).unsqueeze(0)

        slm1 = self._get(
            all_formulas, id1, t_shift1+t_shift, t_span1+t_span
        )
        slm2 = self._get(
            all_formulas, id2, t_shift2+t_shift, t_span2+t_span
        )
        sm = torch.cat([slm1[0], slm2[0]], dim=1)

        _, re = self.lstm_cell(current_ip.expand([-1,sm.shape[1],-1]), (
            sm,  # sm
            torch.cat([slm1[1], slm2[1]], dim=1)   # lm
        ))
        return re


    def forward(self, input_state: torch.Tensor):
        b, _, _ = input_state.shape

        x = input_state

        means, stds, values = [], [], []
        discrete_probs_f_type = []
        discrete_probs_ip_id1 = []
        discrete_probs_ip_id2 = []
        for i in range(b):
            ip:torch.Tensor = x[i]
            f = Formula(buf_formula=ip)
            state_l = DefaultValues.formula_profile_length - f.num_of_dummy
            hidden_s_valid_dim = self.total_amount_of_factors + state_l
            sm , _ = self._get(
                all_formulas= torch.LongTensor(f.std_formula).to(DefaultValues.device),
                idx=torch.tensor([state_l - 1],device=DefaultValues.device),
                t_shift=torch.tensor([0],device=DefaultValues.device),
                t_span=torch.tensor([1],device=DefaultValues.device)
            )
            # sm = [ num_layers, b, hidden_size ]
            out, _ = self.bilstm_summerizer(sm[-1].unsqueeze(1))
            out = out[-1]
            # action output
            fn_types_probs = F.softmax(self.fn_types_output(out), dim=-1)

            prob_valid_dim = hidden_s_valid_dim - 1
            z = torch.zeros([1, self.ip_idx_dim - prob_valid_dim], device=DefaultValues.device)

            ip_idx1_probs = self.ip1_idx_fc(out)[:,:prob_valid_dim]

            ip_idx2_probs = self.ip2_idx_fc(out)[:,:prob_valid_dim]

            ip_idx1_probs = torch.cat(
                [ip_idx1_probs.softmax(dim=-1), z], dim=-1)
            ip_idx2_probs = torch.cat(
                [ip_idx2_probs.softmax(dim=-1), z], dim=-1)
            fn_para_mean = torch.round(
                torch.log(
                    self.fn_para_mean_output(out).relu()+1
                )*self.para_scale-0.5+1e-6
            )
            fn_para_std = torch.clamp(
                self.fn_para_std_output(out), min=1e-2)

            discrete_probs_f_type.append(fn_types_probs)
            discrete_probs_ip_id1.append(ip_idx1_probs)
            discrete_probs_ip_id2.append(ip_idx2_probs)
            means.append(fn_para_mean)
            stds.append(fn_para_std)

        return (
            [
                torch.cat(discrete_probs_f_type, dim=0),
                torch.cat(discrete_probs_ip_id1, dim=0),
                torch.cat(discrete_probs_ip_id2, dim=0)
            ],
            (torch.cat(means,dim=0),torch.cat(stds,dim=0))
        )

    def save(self, name: str):
        torch.save(self.state_dict(), os.path.join(self.save_path, name + ".pth"))

    def load(self, name: str):
        self.load_state_dict(torch.load(os.path.join(self.save_path, name + ".pth")),strict=False)



























class LSTM_p_crit(nn.Module):
    def __init__(
            self,
            total_amount_of_factors: int = DefaultValues.total_amount_of_factors,
            total_types_of_fn: int = DefaultValues.total_types_of_fn,
            formula_profile_length: int = DefaultValues.formula_profile_length,
            lstm_hidden_size: int = 128,
            lstm_num_layers:int = 3,
            save_path:str = get_model_state_dict_path("LSTM_p_crit")

    ):
        self.save_path = save_path
        super().__init__()
        self.total_amount_of_factors = total_amount_of_factors
        # discrete_probs, (mean, std), value = self.policy(state)
        self.para_scale = (formula_profile_length+total_amount_of_factors)
        #all EBD
        self.factors_embedding_l = nn.Sequential(
            nn.Hardtanh(min_val=0-0.5,max_val=total_amount_of_factors-0.5),
            nn.Embedding(total_amount_of_factors, lstm_hidden_size*lstm_num_layers)
        )
        self.factors_embedding_s = nn.Sequential(
            nn.Hardtanh(min_val=0-0.5,max_val=total_amount_of_factors-0.5),
            nn.Embedding(total_amount_of_factors, lstm_hidden_size*lstm_num_layers)
        )
        self.fn_embedding = nn.Sequential(
            nn.Hardtanh(min_val=0 - 0.5, max_val=total_types_of_fn - 0.5),
            nn.Embedding(total_types_of_fn, lstm_hidden_size)
        )

        self.formula_profile_length = formula_profile_length
        self.fn_profile_fuser = nn.Sequential(
            nn.Linear(lstm_hidden_size+2, lstm_hidden_size*2),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size*2, lstm_hidden_size)
        )
        """
        self.sm_fuser = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, lstm_hidden_size)
        )
        self.lm_fuser = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, lstm_hidden_size)
        )
        """

        """
        self.lstm_cell = nn.LSTMCell(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
        )
        """
        self.lstm_cell = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=0.2
        )

        self.bilstm_summerizer = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=0.2,
            bidirectional=True
        )
        self.lstm_num_layers = lstm_num_layers


        # output
        self.value_output = nn.Sequential(
            nn.Linear(2*lstm_hidden_size,lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, 1)
        )


        self.lstm_hidden_size = lstm_hidden_size
        self.drop = nn.Dropout(0.1)

        """hsb = torch.zeros(
            [self.total_amount_of_factors + DefaultValues.formula_profile_length, self.lstm_hidden_size]
        ).to(DefaultValues.device)
        self.register_buffer('hidden_states_buffer', hsb )"""
        self.hidden_states_buffer = None



        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize embeddings with uniform distribution
        for embedding in [self.factors_embedding_l, self.factors_embedding_s, self.fn_embedding]:
            nn.init.uniform_(embedding[1].weight, -0.1, 0.1)  # embedding[1] because it's wrapped in Sequential

        # Initialize all LSTM weights
        for name, param in self.lstm_cell.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 (common practice for LSTMs)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        for name, param in self.bilstm_summerizer.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 (common practice for LSTMs)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

        # Initialize output layers with smaller weights
        for layer in self.value_output:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def _get(self, all_formulas, idx, t_shift, t_span): #num_layers, b, hidden_size
        if idx.detach().cpu().item() < 0:

            #print(f"EBD calling factors_embedding with input = {-idx.detach().cpu().item()-1}")
            return (
                self.factors_embedding_s(-idx-1).view(self.lstm_num_layers,1,self.lstm_hidden_size),
                self.factors_embedding_l(-idx-1).view(self.lstm_num_layers,1,self.lstm_hidden_size)
            )
        type_idx, id1, t_shift1, t_span1, id2, t_shift2, t_span2 = all_formulas[idx.item()].split(1)

        if type_idx.item() == DefaultValues.getter_fn_idx:
            return self._get(all_formulas, id1, t_shift1, t_span1)
        #print(f"EBD calling fn_embedding with input = {type_idx.detach().cpu().item()}")
        current_ip = torch.cat([self.fn_embedding(type_idx), t_shift.unsqueeze(0), t_span.unsqueeze(0)], dim=-1)
        current_ip:torch.Tensor = self.fn_profile_fuser(current_ip).unsqueeze(0)

        slm1 = self._get(
            all_formulas, id1, t_shift1+t_shift, t_span1+t_span
        )
        if type_idx.item() in DefaultValues.fn_of_2_ip:
            slm2 = self._get(
                all_formulas, id2, t_shift2+t_shift, t_span2+t_span
            )
            sm = torch.cat([slm1[0], slm2[0]], dim=1)

            _, re = self.lstm_cell(current_ip.expand([-1,sm.shape[1],-1]), (
                sm,  # sm
                torch.cat([slm1[1], slm2[1]], dim=1)   # lm
            ))
        else:
            _, re = self.lstm_cell(current_ip.expand([-1,slm1[0].shape[1],-1]), slm1)
        return re


    def forward(self, input_state: torch.Tensor):
        b, _, _ = input_state.shape

        x = input_state

        means, stds, values = [], [], []
        discrete_probs_f_type = []
        discrete_probs_ip_id1 = []
        discrete_probs_ip_id2 = []
        for i in range(b):
            ip:torch.Tensor = x[i]
            f = Formula(buf_formula=ip)
            state_l = DefaultValues.formula_profile_length - f.num_of_dummy
            hidden_s_valid_dim = self.total_amount_of_factors + state_l
            sm , _ = self._get(
                all_formulas= torch.LongTensor(f.std_formula).to(DefaultValues.device),
                idx=torch.tensor([state_l - 1],device=DefaultValues.device),
                t_shift=torch.tensor([0],device=DefaultValues.device),
                t_span=torch.tensor([1],device=DefaultValues.device)
            )
            # sm = [ num_layers, b, hidden_size ]
            out, _ = self.bilstm_summerizer(sm[-1].unsqueeze(1))
            out = out[-1]
            value = self.value_output(out)
            values.append(value)

        return torch.cat(values,dim=0)

    def save(self, name: str):
        torch.save(self.state_dict(), os.path.join(self.save_path, name + ".pth"))

    def load(self, name: str):
        self.load_state_dict(torch.load(os.path.join(self.save_path, name + ".pth")),strict=False)























