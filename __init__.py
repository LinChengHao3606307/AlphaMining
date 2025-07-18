
"""
    项目的统一参数
"""

from dataclasses import dataclass
import os
import numpy as np
import torch


@dataclass
class DefaultValues:
    fn_profile_length = 7 #这个不支持修改
    total_types_of_fn = 22 + 1 + 1
    getter_fn_idx = 9
    formula_profile_length = 5
    total_amount_of_formulas = 1
    total_amount_of_factors = 31
    fn_of_2_ip = [3,4,5,6,7,8,21,22]
    buf_formula_shape = (formula_profile_length,
                         fn_profile_length)
    tpv_formula_shape = (2 ** (formula_profile_length+1),
                         5) # output_mask, valid_pos_mask, fn_type, fn_time_shift, fn_time_span
    pse_formula_shape = (formula_profile_length,  # 公式数量
                         (2 * formula_profile_length + 1) + 1, # (2 * formula_profile_length + 1) = 树节点的最大值
                         4) # <tree_pos, fn_type, fn_time_shift, fn_time_span>
    fn_time_para_scale = 10
    device = "cuda"
    project_root_path = os.path.dirname(__file__)
    model_state_dict_path = os.path.join(project_root_path, "main", "Agent", "models_state_dict")

def get_model_state_dict_path(item_name:str):
    return os.path.join(DefaultValues.model_state_dict_path, item_name)

import numpy as np


class Formula:
    """
    公式类，支持三种不同的数据格式：
    
    1. std_formula (标准格式):
       格式: numpy.ndarray, shape=(n, 7), dtype=int32
       每行包含: [type_idx, input1_idx, time_shift1, time_span1, input2_idx, time_shift2, time_span2]
       - type_idx: 函数类型索引 (0-23)
       - input1_idx: 第一个输入的位置索引 (可以是负数表示因子，正数表示公式位置)
       - time_shift1: 第一个输入的时间偏移 (≤0)
       - time_span1: 第一个输入的时间跨度 (≥1)
       - input2_idx: 第二个输入的位置索引 (仅双输入函数使用)
       - time_shift2: 第二个输入的时间偏移 (≤0)
       - time_span2: 第二个输入的时间跨度 (≥1)
       
    2. buf_formula (缓冲区格式):
       格式: numpy.ndarray, shape=(5, 7), dtype=int32
       用于模型输入的固定长度格式，包含：
       - 有效公式数据 + dummy填充
       - 数值已转换：type_idx+1, input_idx+31, time_shift取反, time_span-1
       - dummy行：type_idx=9, input_idx=pos-1, time_shift=0, time_span=1
       
    3. sfn_formula (单函数数值区分型):
       格式: list of dict, 每个dict包含：
       - 'discrete': [type_idx+1, input1_idx+31, input2_idx+31]
       - 'continuous': [-time_shift1, time_span1-1, -time_shift2, time_span2-1]
       
    4. tpv_formula (树位置向量格式):
       格式: numpy.ndarray, shape=(32, 5), dtype=int32
       用于Transformer模型的序列化表示：
       - 第0列: 填充掩码 (1表示填充)
       - 第1列: 有效掩码 (1表示有效)
       - 第2列: 函数类型 (1-31为因子, 32-55为函数)
       - 第3列: 时间偏移
       - 第4列: 时间跨度
    """
    def __init__(self, std_formula=None, sfn_formula=None, buf_formula=None, tpv_formula=None):

        self.std_formula = np.zeros((0, DefaultValues.fn_profile_length), dtype=np.int32)
        self.sfn_formula = []
        self.buf_formula = np.zeros( DefaultValues.buf_formula_shape, dtype=np.int32 )
        self.tpv_formula = np.zeros( DefaultValues.tpv_formula_shape, dtype=np.int32 )
        self.pse_formula = np.zeros( DefaultValues.pse_formula_shape, dtype=np.int32 )
        self.tpv_len = DefaultValues.tpv_formula_shape[0]
        self.num_of_dummy = None
        if not std_formula is None:
            self.std_formula = std_formula
        if not  sfn_formula is None:
            self.set_sfn(sfn_formula)
        elif not buf_formula is None:
            self.set_buf(buf_formula)
        elif not tpv_formula is None:
            self.set_tpv(buf_formula)
        else:
            self.set_std(self.std_formula)

    def get(self,type:str, want_size=False):

        if type == "buf":
            if not want_size:
                return self.buf_formula
            else:
                return DefaultValues.buf_formula_shape
        if type == "tpv":
            if not want_size:
                return self.tpv_formula
            else:
                return DefaultValues.tpv_formula_shape
        if type == "pse":
            if not want_size:
                return self.pse_formula
            else:
                return DefaultValues.pse_formula_shape
        raise NotImplementedError(type + " is not implemented")

    def get_reward_projection_chain(self):
        i = self.std_formula.shape[0] - 1
        if i<0:
            return np.array([0, 0])
        ty, i1, tf1, ts1, i2, tf2, ts2 = self.std_formula[i]
        if i1<0:
            i1 = i
        if i2<0 or ( not (ty in DefaultValues.fn_of_2_ip)):
            i2 = i
        return np.array([i1-i, i2-i])

    def show(self,idx:list[int]=None):
        print("*===    " * 7)
        i = idx
        if i is None:
            i = range(len(self.sfn_formula))
        for pos in i:
            for num in self.std_formula[pos]:
                print(num,end="\t\t")
            print()
        print("*===    " * 7)


    def _std_vali_check(self):
        """批量验证标准格式的合法性"""
        n = self.std_formula.shape[0]
        if n == 0:
            return

        # 批量验证基本维度
        assert n <= DefaultValues.formula_profile_length
        assert self.std_formula.shape[1] == DefaultValues.fn_profile_length

        # 批量验证type范围
        assert np.all(
            (-1 <= self.std_formula[:, 0]) & np.all((self.std_formula[:, 0] < DefaultValues.total_types_of_fn)))

        # 生成索引矩阵用于批量验证
        idx_matrix = np.arange(n) #[:, np.newaxis]

        # 验证输入索引范围
        i1 = self.std_formula[:, 1]
        i2 = self.std_formula[:, 4]
        valid_i1 = (-DefaultValues.total_amount_of_factors <= i1) & (i1 < DefaultValues.formula_profile_length)
        valid_i2 = (-DefaultValues.total_amount_of_factors <= i2) & (i2 < DefaultValues.formula_profile_length)
        assert np.all(valid_i1 & valid_i2)

        # 验证引用顺序 (i1 < current_idx, i2 < current_idx)
        assert np.all((i1 < idx_matrix) & (i2 < idx_matrix))

        # 验证时间参数
        tf1 = self.std_formula[:, 2]
        ts1 = self.std_formula[:, 3]
        tf2 = self.std_formula[:, 5]
        ts2 = self.std_formula[:, 6]
        assert np.all((tf1 <= 0) & (ts1 >= 1) & (tf2 <= 0) & (ts2 >= 1))

    def append_std_fn(self, new_fn):
        """追加标准格式函数"""
        new_fn = np.asarray(new_fn, dtype=np.int32).reshape(1, -1)
        if self.std_formula.shape[0] >= DefaultValues.formula_profile_length:
            raise ValueError(f"Cannot exceed max formula length {DefaultValues.formula_profile_length}")

        self.std_formula = np.vstack([self.std_formula, new_fn])
        self._std_vali_check()
        self._to_sfn()
        self._to_buf()
        self._to_tpv()
        self._to_pse()

    def append_buf_fn(self, new_fn):
        """追加buffer格式函数"""
        new_fn = np.asarray(new_fn, dtype=np.int32).reshape(1, -1)
        if self.num_of_dummy == 0:
            raise ValueError("No space left in buffer")
        self.buf_formula[-self.num_of_dummy] = new_fn
        self.num_of_dummy -= 1

        self._from_buf()
        self._std_vali_check()
        self._to_sfn()
        self._to_tpv()
        self._to_pse()

    def append_sfn_fn(self, new_fn):
        """追加单fn数值区分型函数"""
        if len(self.sfn_formula) >= DefaultValues.formula_profile_length:
            raise ValueError(f"Cannot exceed max formula length {DefaultValues.formula_profile_length}")

        self.sfn_formula.append({
            'discrete': new_fn['discrete'],
            'continuous': new_fn['continuous']
        })
        self._from_sfn()
        self._std_vali_check()
        self._to_buf()
        self._to_tpv()
        self._to_pse()

    def _handle_pse_from_std_at(self, current_std_idx:int, current_tp:int, current_abs_root_time_shift:int, current_time_span:int, level:int):
        if current_std_idx<0:

            self.pse_formula[level, self.tail_idx, 0] = current_tp-1  # tree pos
            self.pse_formula[level, self.tail_idx, 1] = DefaultValues.total_amount_of_factors + current_std_idx  # fn type
            self.pse_formula[level, self.tail_idx, 2] = 0  # time shift
            self.pse_formula[level, self.tail_idx, 3] = 0  # time span
            self.tail_idx += 1
            return
        ty, i1, tf1, ts1, i2, tf2, ts2 = self.std_formula[current_std_idx]

        self.pse_formula[level, self.tail_idx, 0] = current_tp-1 # tree pos
        self.pse_formula[level, self.tail_idx, 1] = ty + 1 + DefaultValues.total_amount_of_factors # fn type
        self.pse_formula[level, self.tail_idx, 2] = -current_abs_root_time_shift  # time shift
        self.pse_formula[level, self.tail_idx, 3] = current_time_span - 1  # time span
        self.tail_idx += 1
        self._handle_pse_from_std_at(i1, current_tp * 2, current_abs_root_time_shift + tf1, ts1, level)
        if ty in DefaultValues.fn_of_2_ip:
            self._handle_pse_from_std_at(i2, current_tp * 2 + 1, current_abs_root_time_shift + tf2, ts2, level)

    def _to_pse(self):
        self.pse_formula *= 0
        self.tail_idx = 0
        length = self.std_formula.shape[0]
        for i in range(length):
            self.tail_idx = 1
            self._handle_pse_from_std_at(i, 1, 0, 1, i)
            m, p = self.pse_formula[i, 1:self.tail_idx, :], self.pse_formula[i, self.tail_idx:, :]
            self.pse_formula[i, 1:, :] = np.concatenate((m[::-1,:],p), axis=0)
            self.pse_formula[i, 0, :] = self.tail_idx - 1
        del self.tail_idx

    def _handle_std_from_pse_at(self, prev_abs_root_time_shift:int):
        current_pos = self.tail_idx
        if self.pse_formula[current_pos][0] == 0:
            return (-1, 0, 1)
        self.tail_idx += 1
        ty, current_abs_shift, current_span = self.pse_formula[current_pos][-3:]
        ty -= DefaultValues.total_amount_of_factors
        if ty < 0:
            return (ty, prev_abs_root_time_shift - current_abs_shift, current_span)
        ty -= 1
        i1, tf1, ts1 = self._handle_std_from_pse_at(current_abs_shift)
        i2, tf2, ts2 = self._handle_std_from_pse_at(current_abs_shift)
        std_pos = self.std_formula.shape[0]-current_pos-1
        self.std_formula[std_pos] = np.array([ty, i1, tf1, ts1, i2, tf2, ts2])
        return (std_pos, prev_abs_root_time_shift - current_abs_shift, current_span+1)

    def _from_pse(self):
        assert False
        length = sum(np.maximum(self.pse_formula[:,0], 0))
        self.std_formula = np.zeros([length,DefaultValues.fn_profile_length])
        self._handle_std_from_pse_at(0)
        self.tail_idx = 0
        del self.tail_idx

    def _handle_tpv_from_std_at(self, current_std_idx:int, current_tp:int, current_abs_root_time_shift:int, current_time_span:int):
        reverse_tp = -current_tp-1
        if current_std_idx < 0:
            self.tpv_formula[reverse_tp][1] = 1  # valid mask
            self.tpv_formula[reverse_tp][2] = DefaultValues.total_amount_of_factors + current_std_idx  # fn type
            self.tpv_formula[reverse_tp][3] = 0  # time shift
            self.tpv_formula[reverse_tp][4] = 0  # time span
            return
        ty, i1, tf1, ts1, i2, tf2, ts2 = self.std_formula[current_std_idx]
        self.tpv_formula[reverse_tp][1] = 1 # valid mask
        self.tpv_formula[reverse_tp][2] = ty + 1 + DefaultValues.total_amount_of_factors # fn type
        self.tpv_formula[reverse_tp][3] = -current_abs_root_time_shift # time shift
        self.tpv_formula[reverse_tp][4] = current_time_span - 1 # time span
        self._handle_tpv_from_std_at(i1, current_tp * 2, current_abs_root_time_shift + tf1, ts1)
        if ty in DefaultValues.fn_of_2_ip:
            self._handle_tpv_from_std_at(i2, current_tp * 2 + 1, current_abs_root_time_shift + tf2, ts2)

    def _to_tpv(self):
        """标准格式 -> 树位置向量"""
        self.tpv_formula *= 0
        dfl = DefaultValues.formula_profile_length
        self.tpv_formula[:self.tpv_len - dfl + self.std_formula.shape[0], 0] = 1
        self._handle_tpv_from_std_at(self.std_formula.shape[0]-1, 1, 0, 1)

    def _handle_std_from_tpv_at(self, current_std_idx:int, current_tp:int,prev_abs_root_time_shift:int):
        reverse_tp = -current_tp - 1
        if self.tpv_formula[reverse_tp][1] == 0:
            return (-1, 0, 1) # dummy
        ty = self.tpv_formula[reverse_tp][2] - DefaultValues.total_amount_of_factors
        this_abs_shift = self.tpv_formula[reverse_tp][3]
        if ty < 0:
            return (-ty, prev_abs_root_time_shift - this_abs_shift, 1) # factor getter
        ty -= 1
        self.tail_idx -= 1
        i1, tf1, ts1 = self._handle_std_from_tpv_at(self.tail_idx, 2*current_tp  , this_abs_shift)
        i2, tf2, ts2 = self._handle_std_from_tpv_at(self.tail_idx, 2*current_tp+1, this_abs_shift)
        self.std_formula[current_std_idx] = np.array([ty, i1, tf1, ts1, i2, tf2, ts2])
        return (current_std_idx, prev_abs_root_time_shift - this_abs_shift, self.tpv_formula[reverse_tp][4]+1)

    def _from_tpv(self):
        assert False
        length = sum(self.tpv_formula[:,0]) - DefaultValues.total_amount_of_factors
        self.tail_idx = length-1
        self.std_formula = np.zeros([length, DefaultValues.fn_profile_length])
        self._handle_std_from_tpv_at(self.tail_idx, 1, 0)
        del self.tail_idx

    def _to_sfn(self):
        """标准格式 -> 单fn数值区分型"""
        n = self.std_formula.shape[0]
        sfn = []

        for i in range(n):
            ty, i1, tf1, ts1, i2, tf2, ts2 = self.std_formula[i]

            # 转换离散参数
            discrete = np.array([
                ty + 1,  # type_idx映射
                i1 + DefaultValues.total_amount_of_factors,  # 输入索引映射
                i2 + DefaultValues.total_amount_of_factors
            ])

            # 转换连续参数
            continuous = np.array([
                -tf1,  # 时间偏移取反
                ts1 - 1,  # 时间跨度-1
                -tf2,
                ts2 - 1
            ])

            sfn.append({'discrete': discrete, 'continuous': continuous})

        self.sfn_formula = sfn
        return sfn

    def _from_sfn(self):
        """单fn数值区分型 -> 标准格式"""
        std = []

        for f in self.sfn_formula:
            discrete = f['discrete']
            continuous = f['continuous']

            # 逆向转换
            ty = discrete[0] - 1
            i1 = discrete[1] - DefaultValues.total_amount_of_factors
            i2 = discrete[2] - DefaultValues.total_amount_of_factors
            tf1 = -continuous[0]
            ts1 = continuous[1] + 1
            tf2 = -continuous[2]
            ts2 = continuous[3] + 1

            std.append([ty, i1, tf1, ts1, i2, tf2, ts2])

        self.std_formula = np.array(std, dtype=np.int32)
        return self.std_formula

    def _to_buf(self):
        """标准格式 -> buffer存储型"""
        n = self.std_formula.shape[0]
        buf = np.full((DefaultValues.formula_profile_length, DefaultValues.fn_profile_length),
                      DefaultValues.getter_fn_idx, dtype=np.int32)

        # 复制有效内容
        if n > 0:
            buf[:n] = self.std_formula
        self.num_of_dummy = DefaultValues.formula_profile_length-n
        # 从后向前填充dummy
        for i in range(n, DefaultValues.formula_profile_length):
            buf[i] = self._get_dummy_np(i)

        # 数值转换
        buf[:, 0] += 1  # type_idx
        buf[:, 1] += DefaultValues.total_amount_of_factors  # 输入索引
        buf[:, 4] += DefaultValues.total_amount_of_factors
        buf[:, [2, 5]] *= -1  # 时间偏移取反
        buf[:, [3, 6]] -= 1  # 时间跨度-1

        self.buf_formula = buf
        return buf

    def _from_buf(self):
        """buffer存储型 -> 标准格式"""
        # 转换数值
        buf = self.buf_formula.copy()
        buf[:, 0] -= 1
        buf[:, 1] -= DefaultValues.total_amount_of_factors
        buf[:, 4] -= DefaultValues.total_amount_of_factors
        buf[:, [2, 5]] *= -1
        buf[:, [3, 6]] += 1
        # 找出有效公式的截止位置
        cutoff = 0
        for i in reversed(range(DefaultValues.formula_profile_length)):
            if not Formula.is_dummy(buf[i],i):
                cutoff = i + 1
                break

        # 截取有效部分
        self.std_formula = buf[:cutoff]
        self.num_of_dummy = DefaultValues.formula_profile_length - cutoff
        return self.std_formula

    @staticmethod
    def is_dummy(fn, pos):
        """判断是否为dummy填充函数"""
        return (
                fn[0] == DefaultValues.getter_fn_idx and
                fn[1] == pos-1 and
                fn[2] == 0 and
                fn[3] == 1 and
                fn[4] == pos-1 and
                fn[5] == 0 and
                fn[6] == 1
        )

    # 统一设置方法
    def set_std(self, arr):
        self.std_formula = arr
        self._std_vali_check()
        self._to_sfn()
        self._to_buf()
        self._to_tpv()
        self._to_pse()

    def set_buf(self, arr:torch.Tensor):
        if type(arr) == torch.Tensor:
            arr = arr.cpu().numpy()
        self.buf_formula = arr
        self._from_buf()
        self._std_vali_check()
        self._to_sfn()
        self._to_tpv()
        self._to_pse()

    def set_sfn(self, sfn):
        self.sfn_formula = sfn
        self._from_sfn()
        self._std_vali_check()
        self._to_buf()
        self._to_tpv()
        self._to_pse()

    def set_tpv(self, tpv):
        self.tpv_formula = tpv
        self._from_tpv()
        self._std_vali_check()
        self._to_buf()
        self._to_sfn()
        self._to_tpv()

    def set_pse(self, pse):
        self.pse_formula = pse
        self._from_pse()
        self._std_vali_check()
        self._to_buf()
        self._to_sfn()

    def __len__(self):
        return len(self.sfn_formula)

    def __getitem__(self, idx):
        return Formula(np.expand_dims(self.std_formula[idx], axis=0))

    def get_dummy(self,i:int = 0):
        return Formula(
            std_formula=self._get_dummy_np(i)
        )

    def _get_dummy_np(self,i:int = 0):
        return np.array([(
            DefaultValues.getter_fn_idx,
            i - 1,
            0, 1,
            i - 1,
            0, 1  )])






