
import os.path


from __init__ import DefaultValues


assert DefaultValues.fn_profile_length == 7


import torch
import torch.nn as nn




class Value_est(nn.Module):
    def __init__(
            self,
            main_model:nn.Module,
            discrete_dims:list[int] = None,
            continuous_dim:int = 4,
            mid_size:int = 2048,
            save_path:str =  DefaultValues.model_state_dict_path + "value_est"

    ):
        self.save_path = save_path
        super().__init__()
        self.main_model = main_model
        if discrete_dims is None:
            self.discrete_dims = [DefaultValues.total_types_of_fn,
                                       DefaultValues.formula_profile_length + DefaultValues.total_amount_of_factors,
                                       DefaultValues.formula_profile_length + DefaultValues.total_amount_of_factors]
        else:
            self.discrete_dims = discrete_dims
        self.continuous_dim = continuous_dim
        self.dim_sum = sum(self.discrete_dims)+continuous_dim*2
        # output
        self.value_output = nn.Sequential(
            nn.Linear(self.dim_sum,mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, 1)
        )



        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.value_output:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, state_input:torch.Tensor):
        dis_input, con_input = self.main_model(state_input)
        flat_op = torch.cat(dis_input+list(con_input),dim=1)
        return self.value_output(flat_op)

    def save(self, name: str):
        torch.save(self.state_dict(), os.path.join(self.save_path, name + ".pth"))

    def load(self, name: str):
        self.load_state_dict(torch.load(os.path.join(self.save_path, name + ".pth")),strict=False)























