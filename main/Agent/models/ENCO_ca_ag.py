
import os.path

import torch
import torch.nn as nn
from numpy.ma.core import shape

from __init__ import DefaultValues, Formula, get_model_state_dict_path
import torch.nn.functional as F

assert DefaultValues.fn_profile_length == 7

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math


class AutoMaskTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, seq_length, dim_feedforward=2048, dropout=0.1, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 创建 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # 使用自定义的 Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.causal_mask = torch.triu(
            torch.ones(seq_length, seq_length) * float('-inf'),
            diagonal=0
        ).unsqueeze(0)

    def forward(self, src, mask):
        """
        参数:
            src: [batch_size, seq_length, d_model] 输入序列
            mask: [batch_size, seq_length] 注意力mask，1表示可以被注意到，0表示不能
        返回:
            output: [batch_size, seq_length, d_model] 编码后的序列
        """
        batch_size, seq_length, _ = src.size()
        # 创建因果注意力mask (确保位置t只能看到n < t的位置)


        # 创建可被注意到的位置mask
        # 首先扩展mask到 [batch_size, seq_length, seq_length]
        # 使得对于每个query位置t，只有mask值为1的key位置n可以被注意到
        expanded_mask = mask.unsqueeze(1).expand(-1, seq_length, -1)

        # 结合因果mask和输入mask
        # 对于位置t，只有n < t且mask[n] == 1的位置可以被注意到
        combined_mask = self.causal_mask.expand(batch_size, -1, -1) + (1.0 - expanded_mask) * float('-inf')

        # 通过Transformer编码器处理
        output = self.transformer_encoder(
            src,
            mask=combined_mask
        )

        return output

class ENCO_ca_ag(nn.Module):
    def __init__(
            self,
            total_amount_of_factors: int = DefaultValues.total_amount_of_factors,
            total_types_of_fn: int = DefaultValues.total_types_of_fn,
            formula_profile_length: int = DefaultValues.formula_profile_length,
            d_model: int = 128,
            num_layers:int = 2,
            nhead:int = 8,
            save_path:str = DefaultValues.model_state_dict_path + "ENCO_ca_ag"

    ):
        assert d_model % nhead == 0
        self.save_path = save_path
        super().__init__()
        self.total_amount_of_factors = total_amount_of_factors
        # discrete_probs, (mean, std), value = self.policy(state)
        self.para_scale = (formula_profile_length+total_amount_of_factors)
        #all EBD
        self.embedding = nn.Embedding(total_amount_of_factors+total_types_of_fn, d_model)
        self.log_seq_len = formula_profile_length+1
        self.positional_ebd = nn.Parameter(torch.randn(
            [1, 2**self.log_seq_len, d_model]
        ))
        self.time_para_fc = nn.Linear(2, d_model,bias=False)

        self.transformer_encoders = [
            AutoMaskTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                seq_length=2**(self.log_seq_len-i),
                dim_feedforward=4*d_model,
                num_layers=num_layers
            )
            for i in range(self.log_seq_len)
        ]
        # action output
        self.fn_types_output = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, total_types_of_fn)
        )
        self.ip_idx_dim = DefaultValues.formula_profile_length + DefaultValues.total_amount_of_factors


        self.ip1_idx_fc = nn.Sequential(
            nn.Linear(d_model, 2 * self.ip_idx_dim),
            nn.ReLU(),
            nn.Linear(2 * self.ip_idx_dim, self.ip_idx_dim)
        )
        self.ip2_idx_fc = nn.Sequential(
            nn.Linear(d_model, 2 * self.ip_idx_dim),
            nn.ReLU(),
            nn.Linear(2 * self.ip_idx_dim, self.ip_idx_dim)
        )



        self.fn_para_mean_output = nn.Linear(d_model, DefaultValues.fn_profile_length - 3)
        self.fn_para_std_output = nn.Linear(d_model, DefaultValues.fn_profile_length - 3)

        self.lstm_hidden_and_econ_dim = d_model
        self.drop = nn.Dropout(0.1)

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
        for embedding in [self.embedding]:
            nn.init.uniform_(embedding.weight, -0.1, 0.1)  # embedding[1] because it's wrapped in Sequential

        # Initialize encoder weights
        def _init_transformer_weights(module):
            # 初始化自注意力层
            if isinstance(module, nn.MultiheadAttention):
                # Query, Key, Value projections
                nn.init.kaiming_normal_(module.in_proj_weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.in_proj_bias, 0.)

                # Output projection
                nn.init.xavier_normal_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0.)

            # 初始化前馈网络层
            elif isinstance(module, nn.Linear):
                if module._get_name() == 'fc1':
                    # 第一个全连接层使用Kaiming初始化
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # 其他线性层使用Xavier初始化
                    nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)

            # 初始化层归一化
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.)

        # 对每个编码器层应用初始化
        for encoder in self.transformer_encoders:
            for layer in encoder.transformer_encoder.layers:
                layer.apply(_init_transformer_weights)

        # Initialize output layers with smaller weights
        for layer in self.fn_types_output:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        nn.init.uniform_(self.fn_para_mean_output.weight, -0.01, 0.01)
        nn.init.uniform_(self.fn_para_std_output.weight, -0.01, 0.01)


    def forward(self, input_state: torch.Tensor):
        op_mask = input_state[:,:,0] * float('-inf')
        mask = input_state[:,:,1]
        fn_type = input_state[:,:,2]
        fn_time_para = input_state[:,:,3:]
        input_seq = self.positional_ebd.expand([input_state.shape[0],-1,-1]) + \
                    F.normalize(self.embedding(fn_type) + self.time_para_fc(fn_time_para),dim=-1)
        input_seq = torch.cat([
            input_seq[:, :-1, :],
            torch.zeros_like(input_seq[:,-1,:]).unsqueeze(1)
        ],dim=1)
        x = input_seq
        for i in range(self.log_seq_len):
            current_seq_len = 2**(self.log_seq_len-i)
            x = input_seq[:, current_seq_len:, :] + self.transformer_encoders[i](
                x[:, current_seq_len:, :], mask[:, current_seq_len:]
            )
        out = x[:, -1, :]
        fn_types_probs = F.softmax(self.fn_types_output(out), dim=-1)

        ip_idx1_probs = self.ip1_idx_fc(out) + op_mask
        ip_idx2_probs = self.ip2_idx_fc(out) + op_mask

        fn_para_mean = torch.round(
            torch.log(
                self.fn_para_mean_output(out).relu() + 1
            ) * self.para_scale - 0.5 + 1e-6
        )
        fn_para_std = torch.clamp(
            self.fn_para_std_output(out), min=1e-2)
        return (
            [
                fn_types_probs,
                ip_idx1_probs,
                ip_idx2_probs
            ],
            (
                fn_para_mean,
                fn_para_std
            )
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
            value = self.value_output(out)
            values.append(value)

        return torch.cat(values,dim=0)

    def save(self, name: str):
        torch.save(self.state_dict(), os.path.join(self.save_path, name + ".pth"))

    def load(self, name: str):
        self.load_state_dict(torch.load(os.path.join(self.save_path, name + ".pth")),strict=False)























