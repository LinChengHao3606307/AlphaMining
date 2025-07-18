import torch
import torch.nn as nn



class Linear_ac(nn.Module):
    def __init__(
            self,ip_dim: int
    ):
        super().__init__()
        self.ip_dim = ip_dim
        self.para = nn.Parameter(torch.rand([1,self.ip_dim]))

    def forward(self, ip: torch.Tensor):
        w = torch.softmax(self.para, dim=-1)
        return (ip * w).sum(dim=-1).unsqueeze(-1)
