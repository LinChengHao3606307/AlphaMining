
import torch
import torch.nn as nn




def CalRank(tensor:torch.Tensor):
    hard_tanh = nn.Hardtanh(min_val=0.0, max_val=1.0)
    tensor = tensor.unsqueeze(-1)
    tensor = tensor.expand(*tensor.shape[:-1], tensor.shape[-1])
    all_diff = tensor - tensor.transpose(dim0=-2,dim1=-1)
    min_num = torch.min(torch.abs(all_diff))+1e-6
    all_diff:torch.Tensor =  hard_tanh(all_diff/min_num)
    return all_diff.sum(dim=-1)

t = torch.tensor([4,5,2,3,2,8,14,66,22,13])
print(t)
print(CalRank(t))