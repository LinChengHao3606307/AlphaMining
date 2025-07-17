
"""
    为训练下游模型设计的数据集
    FactorComputer组件
        根据公式计算因子值
        _get(self, id: int, t: int)
            取得id位置在时间t的计算结果
            id<0代表直接从原始数据中读取，id>=0则读取前面函数的结果
        _compute(self, type_idx: int,
                 id1: int, t_shift1: int, t_span1: int,
                 id2: int, t_shift2: int, t_span2: int)
            根据type_idx调用不同计算步骤，计算结果
    DataSet
        直接衔接pd.DataFrame，需要其索引为数字，第一列日期，第二列return，后面factor，如
                      date    return  factor_1  ...  factor_38  factor_39  factor_40
            0     20171231  1.764052  0.309724  ...  -0.047507  -0.143494   0.039510
            1     20180101  0.400157 -0.737456  ...   0.057382   0.443630   0.338378
            2     20180102  0.978738 -1.536920  ...  -1.875664  -0.398316  -0.842183
            3     20180103  2.240893 -0.562255  ...  -2.169014   0.259012  -0.049632
            4     20180104  1.867558 -1.599511  ...   0.362283   0.740711  -1.230245
            ...        ...       ...       ...  ...        ...        ...        ...
            4995  20310904 -0.101374 -1.809282  ...  -0.056161   0.271827  -0.907515
            4996  20310905  0.746666  0.042359  ...   0.452369   0.050507  -1.299441
            4997  20310906  0.929182  0.516872  ...  -0.798715   0.667815   2.662423
            4998  20310907  0.229418 -0.032921  ...  -0.633912  -1.021195   0.105044
            4999  20310908  0.414406  1.298111  ...   1.292526  -0.345373  -0.220644
        把原始数据转成np之后，会继续创造类别为公式结果的列，当向DataSet输入公式时，会依据公式填充这些列
    SubSet组件
        用于分割数据集

"""
from __init__ import DefaultValues

import math
import statistics
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class DataSet(Dataset):
    def __init__(self, original_data: pd.DataFrame, num_formulas: int = DefaultValues.total_amount_of_formulas, device:str= DefaultValues.device,eps=1e-6):
        self.device = device
        super().__init__()
        selected_columns = original_data.iloc[:, 2:]
        self.target = original_data.iloc[:, 1].to_numpy()
        self.original_data = selected_columns.to_numpy()
        self.num_formulas = num_formulas + 1
        self.num_days, self.num_col = self.original_data.shape
        # 扩展数组以容纳公式结果
        self.original_data = np.hstack([
            self.original_data,
            np.zeros((self.num_days, self.num_formulas))
        ])
        self.formulas = []
        self.eps = eps

    def reset(self):
        self.original_data[:, self.num_col:] = 0
        self.formulas = []

    def replace_formula(self, new_formula, idx):
        if idx < 0 or idx >= self.num_formulas:
            raise ValueError("Invalid formula index")
        if len(self.formulas) <= idx:
            self.formulas += [None] * (idx - len(self.formulas) + 1)
        self.formulas[idx] = new_formula
        # 使用向量化计算替换逐日循环
        factor_computer = FactorComputer(self.original_data, self.num_days, self.num_col, self.eps)
        factor_computer.set_all_fn(new_formula)
        all_results = factor_computer.compute_vectorized()
        # 将结果写入对应的列
        target_col = self.num_col + idx
        self.original_data[:, target_col] = all_results

    def __getitem__(self, idx):
        ip = self.original_data[idx]
        return torch.FloatTensor([self.target[idx]]).to(self.device), torch.FloatTensor(ip[-self.num_formulas:]).to(self.device)

    def __len__(self):
        return self.num_days

class FactorComputer:
    def __init__(self, data, input_len: int, input_valid_col: int, eps:float=1e-6):
        self.data = data
        self.input_len = input_len
        self.input_valid_col = input_valid_col
        self.all_fn_conf = []
        self.eps = eps

    def set_all_fn(self, all_fn_conf):
        self.all_fn_conf = all_fn_conf

    def compute_vectorized(self, type_idx: int=None,
                 id1: int=None, t_shift1: int=None, t_span1: int=None,
                 id2: int=None, t_shift2: int=None, t_span2: int=None):
        if type_idx is None:
            formula = self.all_fn_conf[-1]
            type_idx = round(formula[0])
            id1, t_shift1, t_span1 = round(formula[1]), round(formula[2]), round(formula[3])
            id2, t_shift2, t_span2 = round(formula[4]), round(formula[5]), round(formula[6])

        assert t_shift1 <= 0 and t_shift2 <= 0, "时间偏移不能为正"
        assert t_span1 >= 1 and t_span2 >= 1, "时间跨度必须至少为1"

        # 处理操作数1
        op1 = self._get_vectorized(id1, t_shift1)
        # 处理操作数2（如果存在）
        op2 = self._get_vectorized(id2, t_shift2) if type_idx in DefaultValues.fn_of_2_ip else None

        # 根据类型执行向量化计算
        if type_idx == 0:  # 常数
            return np.full(self.input_len, (id1 + id2 + t_shift1 + t_shift2 + t_span1 + t_span2) / 6)
        elif type_idx == 1:  # 绝对值
            return np.abs(op1)
        elif type_idx == 2:  # 对数
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(op1 > 0, np.log(op1), 0.0)
        elif type_idx == 3:  # 加法
            return op1 + op2
        elif type_idx == 4:  # 减法
            return op1 - op2
        elif type_idx == 5:  # 乘法
            return op1 * op2
        elif type_idx == 6:  # 除法
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(op1, op2)
                return np.where(np.isfinite(result), result, 0.0)
        elif type_idx == 7:  # 最大值
            return np.maximum(op1, op2)
        elif type_idx == 8:  # 最小值
            return np.minimum(op1, op2)
        elif type_idx == 9:  # 直接引用
            return op1
        elif type_idx == 10:  # 均值
            return pd.Series(op1).rolling(t_span1, min_periods=1).mean().values
        elif type_idx == 11:  # 中位数
            return pd.Series(op1).rolling(t_span1).median().fillna(0).values
        elif type_idx == 12:  # 求和
            return pd.Series(op1).rolling(t_span1).sum().fillna(0).values
        elif type_idx == 13:  # 标准差
            return pd.Series(op1).rolling(t_span1).std().fillna(0).values
        elif type_idx == 14:  # 方差
            return pd.Series(op1).rolling(t_span1).var().fillna(0).values
        elif type_idx == 15:  # 最大值
            return pd.Series(op1).rolling(t_span1).max().fillna(0).values
        elif type_idx == 16:  # 最小值
            return pd.Series(op1).rolling(t_span1).min().fillna(0).values
        elif type_idx == 17:  # 平均绝对偏差
            def mad(x):
                return np.mean(np.abs(x - np.mean(x)))
            return pd.Series(op1).rolling(t_span1).apply(mad, raw=True).fillna(0).values
        elif type_idx == 18:  # Delta
            shifted = pd.Series(op1).shift(t_span1).fillna(0).values
            return op1 - shifted
        elif type_idx == 19:  # WMA
            def wma(series):
                weights = np.arange(1, t_span1 + 1)
                return np.convolve(series, weights, 'valid') / weights.sum()
            wma_vals = np.zeros_like(op1)
            wma_vals[t_span1-1:] = wma(op1)
            return wma_vals
        elif type_idx == 20:  # EMA
            alpha = 2 / (t_span1 + 1)
            ema = pd.Series(op1).ewm(alpha=alpha, adjust=False).mean().values
            return ema
        elif type_idx == 21:  # 协方差
            cov = pd.Series(op1).rolling(t_span1).cov(pd.Series(op2)).fillna(0).values
            return cov
        elif type_idx == 22:  # 相关系数
            corr = pd.Series(op1).rolling(t_span1).corr(pd.Series(op2)).fillna(0).values
            return corr
        else:
            raise ValueError(f"Unsupported type_idx: {type_idx}")

    def _get_vectorized(self, id: int, t_shift: int):
        if id < 0:
            col_idx = -id - 1
            if col_idx >= self.input_valid_col:
                return np.zeros(self.input_len)
            data = self.data[:, col_idx]
        else:
            formula = self.all_fn_conf[id]
            type_idx = round(formula[0])
            id1, t_shift1, t_span1 = round(formula[1]), round(formula[2]), round(formula[3])
            id2, t_shift2, t_span2 = round(formula[4]), round(formula[5]), round(formula[6])
            return self.compute_vectorized(
                type_idx,
                id1, t_shift1+t_shift, t_span1,
                id2, t_shift2+t_shift, t_span2
            )

        # 应用时间偏移
        shifted = np.roll(data, -t_shift)
        shifted[:t_shift] = 0
        return shifted

class SubSet(Dataset):
    def __init__(self,dataset:DataSet, sub_range:tuple[int,int]):
        super().__init__()
        self.dataset = dataset
        self.sub_range = sub_range
        self.l = sub_range[1]-sub_range[0]
        assert self.l > 0
    def __getitem__(self,idx):
        assert self.sub_range[0]+idx<self.sub_range[1]
        return self.dataset[self.sub_range[0]+idx]

    def __len__(self):
        return self.l

class UnionSet(Dataset):
    def __init__(self,datasets:list[Dataset]):
        super().__init__()
        self.datasets = datasets
        self.l = 0
        self.all_l = [0]
        for s in self.datasets:
            ls = len(s)
            self.l += ls
            self.all_l.append(self.l)

    def __getitem__(self,idx):
        set_i = 0
        while self.all_l[set_i+1] <= idx:
            set_i += 1

        return self.datasets[set_i][idx-self.all_l[set_i]]

    def __len__(self):
        return self.l