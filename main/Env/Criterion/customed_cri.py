

"""
    下游模型的损失标准
        GSort组件
            由于序本不可导，为了引导模型形成参数，这里只能强行为这一步骤的导数赋值
            赋值的原则是让导数引导模型形成参数
            赋值的做法是令每个数的导数为
                1 / (1 + distance_above) - 1 / (1 + distance_below)
            其中distance_above是它与排在它正上方的数的差
            其中distance_above是它与排在它正下方的数的差
            此值衡量此数增大带来它自己的排名值的变化
        CustomizedCri
            计算avg的ic，ir，head的ic，ir，worst的ic，ir，其中
                ic是在batch内计算
                ir是在batch间计算
            计算结果为他们以预定权重相加倒转为loss，并翻倍他们中最弱的的loss值作为额外惩罚

"""
import torch.nn as nn

import torch
from torch.autograd import Function

from __init__ import DefaultValues

"""
     [test_loss: 0.000947]______[vali_loss: 0.000542]
     [test_loss: 0.000968]______[vali_loss: 0.000530]
     [test_loss: 0.000981]______[vali_loss: 0.000568]
     [test_loss: 0.000976]______[vali_loss: 0.000572]
     [test_loss: 0.000952]______[vali_loss: 0.000533]
     [test_loss: 0.001095]______[vali_loss: 0.000625]
     [test_loss: 0.000925]______[vali_loss: 0.000519]
     [test_loss: 0.001047]______[vali_loss: 0.000599]
     [test_loss: 0.001019]______[vali_loss: 0.000613]
     [test_loss: 0.001000]______[vali_loss: 0.000600]
     [test_loss: 0.000993]______[vali_loss: 0.000571]
 cur [test_loss: 0.000949]______[vali_loss: 0.000560]
"""
class GSort(Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor):
        upb = torch.max(input)
        lwb = torch.min(input)
        l = input.shape[-1]
        input = input*l /(upb-lwb+1e-6)
        # 排序输入，获取排序后的值和索引
        sorted_values, sorted_indices = torch.sort(input)
        n = input.numel()  # 获取元素总数

        # 计算distance_below和distance_above（相邻差值）
        distance_below = torch.zeros_like(sorted_values)
        distance_above = torch.zeros_like(sorted_values)
        if n > 1:
            diff = torch.softmax(sorted_values[1:] - sorted_values[:-1],dim=-1)
            # distance_below[i] = sorted_values[i] - sorted_values[i-1]，i>0
            distance_below[1:] = diff
            # distance_above[i] = sorted_values[i+1] - sorted_values[i], i < n-1
            distance_above[:-1] = diff

        # 计算梯度项：1/(1+distance_above) - 1/(1+distance_below)
        grad_terms = 1 / (1 + distance_above) - 1 / (1 + distance_below)

        # 将梯度项分配到原输入对应的位置
        grad = torch.zeros_like(input)
        grad.scatter_(0, sorted_indices, grad_terms)  # 使用scatter_进行高效赋值

        ctx.save_for_backward(grad)
        return sorted_indices.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad_output * grad




class CustomizedCri(nn.Module):
    def __init__(self,
                 avg_ic_significance_weight: float = 1,
                 avg_ir_significance_weight: float = 1,

                 head_ic_significance_weight: float = 2,
                 head_ir_significance_weight: float = 2,
                 head_percentage: float = 0.2,

                 worst_ic_significance_weight: float = 0,
                 worst_ir_significance_weight: float = 0,
                 num_of_worst: int = 10,

                 history_length:int = 20,

                 min_over_mean_weight:float = 1
                 ):
        super().__init__()
        # 初始化各指标权重参数
        self.weights = torch.FloatTensor([
            avg_ic_significance_weight,
            avg_ir_significance_weight,
            head_ic_significance_weight,
            head_ir_significance_weight,
            worst_ic_significance_weight,
            worst_ir_significance_weight
        ]).to(DefaultValues.device)
        self.head_percentage = head_percentage
        self.num_of_worst = num_of_worst

        self.ic_history = []
        self.head_ic_history = []
        self.worst_ic_history = []
        self.history_length = history_length

        self.min_over_mean_weight = min_over_mean_weight

    def forward(self, model_op: torch.Tensor, tgt_op: torch.Tensor):
        # 基本维度校验

        assert model_op.shape == tgt_op.shape, "输入张量形状不一致"
        batch = model_op.shape[0]


        model_rank = GSort.apply(model_op.squeeze())
        tgt_rank = GSort.apply(tgt_op.squeeze())


        rank_diff_sq_seq = (model_rank - tgt_rank).pow(2)
        ic_sequence = 1.0 / batch - (6.0 * rank_diff_sq_seq) / (batch * (batch ** 2 - 1))

        # ------------------- 核心指标计算 -------------------
        # 平均IC计算
        avg_ic = ic_sequence.sum()
        avg_ic = avg_ic.unsqueeze(0)

        # 头部样本计算（前20%）
        head_num = int(batch * self.head_percentage)
        _, head_indices = torch.topk(model_op.squeeze(), head_num)
        head_ic = torch.corrcoef(torch.stack([
            GSort.apply(model_op.squeeze()[head_indices]),
            GSort.apply(tgt_op.squeeze()[head_indices])
        ]))[0, 1]
        head_ic = head_ic.unsqueeze(0)


        # 最差样本计算（末位N个）
        _, worst_indices = torch.topk(-model_op.squeeze(), self.num_of_worst)
        worst_ic = torch.corrcoef(torch.stack([
            GSort.apply(model_op.squeeze()[worst_indices]),
            GSort.apply(tgt_op.squeeze()[worst_indices])
        ]))[0, 1] if self.num_of_worst > 1 else 0.0  # 单样本无法计算相关系数
        worst_ic = worst_ic.unsqueeze(0)
        # ------------------- IR 计算 -------------------


        # 计算 IR（IC 的均值 / IC 的标准差）
        if len(self.ic_history) > 1:  # 至少需要两个批次的数据
            avg_ir = (torch.cat(self.ic_history+[avg_ic]).mean()
                                   / torch.cat(self.ic_history+[avg_ic]).std()).unsqueeze(0)
            head_ir = (torch.cat(self.head_ic_history+[head_ic]).mean()
                                    / torch.cat(self.head_ic_history+[head_ic]).std()).unsqueeze(0)
            worst_ir = (torch.cat(self.worst_ic_history+[worst_ic]).mean()
                                     / torch.cat(self.worst_ic_history+[worst_ic]).std()).unsqueeze(0)
        else:
            # 如果只有一个批次的数据，IR 无法计算，暂时设为 0
            avg_ir = torch.tensor([0.0],requires_grad=True).to(DefaultValues.device)
            head_ir = torch.tensor([0.0],requires_grad=True).to(DefaultValues.device)
            worst_ir = torch.tensor([0.0],requires_grad=True).to(DefaultValues.device)

        # 将当前批次的 IC 值存储到历史记录中
        self.ic_history.append(avg_ic.clone().detach())
        self.head_ic_history.append(head_ic.clone().detach())
        self.worst_ic_history.append(worst_ic.clone().detach())
        if len(self.ic_history)>self.history_length:
            self.ic_history = self.ic_history[1:]
            self.head_ic_history = self.head_ic_history[1:]
            self.worst_ic_history = self.worst_ic_history[1:]

        # ------------------- 加权综合损失 -------------------
        result = torch.cat([
            avg_ic,avg_ir,
            head_ic,head_ir,
            worst_ic,worst_ir
        ])*self.weights

        return (
                - torch.min( result) *(self.min_over_mean_weight/(1+self.min_over_mean_weight))
                - torch.mean(result) *(1/(1+self.min_over_mean_weight))
        )

