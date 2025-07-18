
import os.path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from __init__ import DefaultValues
import torch.nn.functional as F
assert DefaultValues.fn_profile_length == 7



class LSTM_ps_ag(nn.Module):
    def __init__(
            self,
            total_amount_of_factors: int = DefaultValues.total_amount_of_factors,
            total_types_of_fn: int = DefaultValues.total_types_of_fn,
            formula_profile_length: int = DefaultValues.formula_profile_length,
            lstm_hidden_size: int = 128,
            enco_num_layers: int = 2,
            lstm_num_layers:int = 1,
            summarizer_lstm_num_layers: int = 4,
            enco_nhead:int=8,
            save_path:str = os.path.join(DefaultValues.model_state_dict_path, "LSTM_ps_ag")

    ):
        self.save_path = save_path
        super().__init__()
        self.total_amount_of_factors = total_amount_of_factors
        self.total_types_of_fn = total_types_of_fn
        # discrete_probs, (mean, std), value = self.policy(state)
        #all EBD
        self.fn_type_ebd = nn.Embedding(self.total_amount_of_factors + self.total_types_of_fn, lstm_hidden_size)
        self.tree_pos_ebd = nn.Embedding(2**(formula_profile_length+1)-1, lstm_hidden_size)
        self.time_para_fc = nn.Linear(2, lstm_hidden_size,bias=False)
        self.lstm_hidden_size = lstm_hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden_size,
            nhead=enco_nhead,
            dim_feedforward=4 * lstm_hidden_size,
            batch_first=True  # 使用[batch, seq, feature]格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enco_num_layers)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.summarizer_lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=summarizer_lstm_num_layers,
            batch_first=True,
            dropout=0.2
        )
        # action output
        self.fn_types_output = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size * 2, total_types_of_fn)
        )
        self.ip_idx_dim = DefaultValues.formula_profile_length + DefaultValues.total_amount_of_factors


        self.ip1_idx_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 2 * self.ip_idx_dim),
            nn.ReLU(),
            nn.Linear(2 * self.ip_idx_dim, self.ip_idx_dim)
        )
        self.ip2_idx_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 2 * self.ip_idx_dim),
            nn.ReLU(),
            nn.Linear(2 * self.ip_idx_dim, self.ip_idx_dim)
        )



        self.fn_para_mean_output = nn.Linear(lstm_hidden_size, DefaultValues.fn_profile_length - 3)
        self.fn_para_std_output = nn.Linear(lstm_hidden_size, DefaultValues.fn_profile_length - 3)
        self.drop = nn.Dropout(0.1)

        # 初始化权重
        self.initialize_weights()
        self.zeros = torch.zeros([1, DefaultValues.total_amount_of_factors]).to(DefaultValues.device)
        self.factor_idx = torch.arange(
            0, DefaultValues.total_amount_of_factors, dtype=torch.long
        ).to(DefaultValues.device).unsqueeze(0)

    def initialize_weights(self):
        # Initialize weights for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize embeddings with uniform distribution
        for embedding in [self.fn_type_ebd,self.tree_pos_ebd]:
            nn.init.uniform_(embedding.weight, -0.1, 0.1)  # embedding[1] because it's wrapped in Sequential

        # Initialize all LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 (common practice for LSTMs)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        for name, param in self.summarizer_lstm.named_parameters():
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


    def forward(self, input_state: torch.Tensor):
        """
            input: [
                b ,
                DefaultValues.formula_profile_length,
                seq_len: 1+real_seq_len ,
                4: <tree_pos, fn_type, fn_time_shift, fn_time_span>
            ]
        """
        b, _formula_profile_length, seq_len, _4 = input_state.shape
        op_mask = input_state[:, :, 0, 0] #>> [b , DefaultValues.formula_profile_length]
        lengths = op_mask.view(b * DefaultValues.formula_profile_length)+1 #>> [b * DefaultValues.formula_profile_length]
        op_mask = torch.where(op_mask > 0, 0.0, float('-inf')) #>> [b , DefaultValues.formula_profile_length]
        op_mask = torch.cat([
            self.zeros.expand(b,-1),
            op_mask
        ],dim=-1)

        input_seq = input_state.view(b * DefaultValues.formula_profile_length, seq_len, 4) #>> [b * DefaultValues.formula_profile_length, seq_len, 4]
        tree_pos = input_seq[:,:,0]
        fn_type = input_seq[:,:,1]
        fn_time_para = input_seq[:,:,2:].float()/DefaultValues.fn_time_para_scale
        input_seq = self.tree_pos_ebd(tree_pos) + \
                    F.normalize(self.fn_type_ebd(fn_type) + self.time_para_fc(fn_time_para),dim=-1)
        input_seq[:,0,:] = 0.01 * torch.rand_like(input_seq[:,0,:]) / ( + lengths.float().unsqueeze(-1))

        # Packing the padded sequence
        pack_data = pack_padded_sequence(input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)

        output, hidden = self.lstm(pack_data)

        # Unpacking the output
        output, _ = pad_packed_sequence(output, batch_first=True)

        # Ensure the index is within the valid range
        valid_indices = torch.arange(output.size(0))

        # Select the last valid output for each sequence
        out = output[valid_indices, lengths - 1].view(b, DefaultValues.formula_profile_length, self.lstm_hidden_size)

        factors = self.fn_type_ebd(self.factor_idx).expand(
            b, DefaultValues.total_amount_of_factors, self.lstm_hidden_size
        )
        out = torch.cat([factors, out],dim=1)
        out = self.transformer_encoder(out)
        out, _ = self.summarizer_lstm(out)
        out = out[:,-1]
        fn_types_probs = F.softmax(self.fn_types_output(out), dim=-1)
        ip_idx1_probs = (self.ip1_idx_fc(out) + op_mask).softmax(dim=-1)
        ip_idx2_probs = (self.ip2_idx_fc(out) + op_mask).softmax(dim=-1)
        fn_para_mean = torch.round(
            torch.log(
                self.fn_para_mean_output(out).relu() + 1
            ) * DefaultValues.fn_time_para_scale - 0.5 + 1e-6
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



























