
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
                         (fn_profile_length-1)//2 + 1 + 1)
    device = "cuda"
    project_root_path = os.path.dirname(__file__)
    model_state_dict_path = os.path.join(project_root_path, "main", "Agent", "models_state_dict")

def get_model_state_dict_path(item_name:str):
    return os.path.join(DefaultValues.model_state_dict_path, item_name)

import numpy as np


class Formula:
    def __init__(self, std_formula=None, sfn_formula=None, buf_formula=None):

        self.std_formula = np.zeros((0, DefaultValues.fn_profile_length), dtype=np.int32)
        self.sfn_formula = []
        self.buf_formula = np.zeros( DefaultValues.buf_formula_shape, dtype=np.int32 )
        self.tpv_formula = np.zeros( DefaultValues.tpv_formula_shape, dtype=np.int32 )
        self.num_of_dummy = None

        if not std_formula is None:
            self.std_formula = std_formula
        if not  sfn_formula is None:
            self.set_sfn(sfn_formula)
        elif not buf_formula is None:
            self.set_buf(buf_formula)
        else:
            self.set_std(self.std_formula)
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
        """print("= "*40)
        print("i1:")
        print(i1)
        print("i2:")
        print(i2)
        print("check with>>")
        print(idx_matrix)"""
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

    def append_sfn_fn(self, new_fn):
        """追加单fn数值区分型函数"""
        if len(self.sfn_formula) >= DefaultValues.formula_profile_length:
            raise ValueError(f"Cannot exceed max formula length {DefaultValues.formula_profile_length}")

        self.sfn_formula.append(new_fn)
        self._from_sfn()
        self._std_vali_check()
        self._to_buf()
        self._to_tpv()

    def _handle_tpv_from_std_at(self, current_std_idx:int, current_tp:int, current_abs_root_time_shift:int, current_time_span:int):
        if current_std_idx<0:
            self.tpv_formula[current_tp][1] = 1  # valid mask
            self.tpv_formula[current_tp][2] = DefaultValues.total_amount_of_factors - current_std_idx  # fn type
            self.tpv_formula[current_tp][3] = 0  # time shift
            self.tpv_formula[current_tp][4] = 0  # time span
            return
        ty, i1, tf1, ts1, i2, tf2, ts2 = self.std_formula[current_std_idx]
        self.tpv_formula[current_tp][1] = 1 # valid mask
        self.tpv_formula[current_tp][2] = ty + 1 + DefaultValues.total_amount_of_factors # fn type
        self.tpv_formula[current_tp][3] = -current_abs_root_time_shift # time shift
        self.tpv_formula[current_tp][4] = current_time_span - 1 # time span
        self._handle_tpv_from_std_at(i1, current_tp * 2, current_abs_root_time_shift + tf1, ts1)
        if ty in DefaultValues.fn_of_2_ip:
            self._handle_tpv_from_std_at(i2, current_tp * 2 + 1, current_abs_root_time_shift + tf2, ts2)

    def _to_tpv(self):
        """标准格式 -> 树位置向量"""
        self.tpv_formula *= 0
        self.tpv_formula[-(DefaultValues.formula_profile_length - self.std_formula.shape[0]):, 0] = 1
        self._handle_tpv_from_std_at(self.std_formula.shape[0]-1, 1, 0, 1)

    def _handle_std_from_tpv_at(self, current_std_idx:int, current_tp:int,prev_abs_root_time_shift:int):
        if self.tpv_formula[current_tp][1] == 0:
            return (-1, 0, 1) # dummy
        ty = self.tpv_formula[current_tp][2] - DefaultValues.total_amount_of_factors
        if ty < 0:
            return (-ty, 0, 1) # factor getter
        ty += 1
        self.tail_idx -= 1
        this_abs_shift = self.tpv_formula[current_tp][3]
        i1, tf1, ts1 = self._handle_tpv_from_std_at(self.tail_idx, 2*current_tp  , this_abs_shift)
        i2, tf2, ts2 = self._handle_tpv_from_std_at(self.tail_idx, 2*current_tp+1, this_abs_shift)
        self.std_formula[current_std_idx] = np.array([ty, i1, tf1, ts1, i2, tf2, ts2])
        return (current_std_idx, prev_abs_root_time_shift - this_abs_shift, self.tpv_formula[current_tp][4]+1)

    def _from_tpv(self):
        length = DefaultValues.formula_profile_length - sum(self.tpv_formula[:,0])
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

    def set_buf(self, arr:torch.Tensor):
        if type(arr) == torch.Tensor:
            arr = arr.cpu().numpy()
        self.buf_formula = arr
        self._from_buf()
        self._std_vali_check()
        self._to_sfn()
        self._to_tpv()

    def set_sfn(self, sfn):
        self.sfn_formula = sfn
        self._from_sfn()
        self._std_vali_check()
        self._to_buf()
        self._to_tpv()

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







import contextlib
import importlib
import os
from pathlib import Path
import pickle
import pkgutil
import re
import sys
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union



def get_module_by_module_path(module_path: Union[str, ModuleType]):
    """Load module path

    :param module_path:
    :return:
    :raises: ModuleNotFoundError
    """
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")

    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            module_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")))
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)
    return module


def split_module_path(module_path: str) -> Tuple[str, str]:
    """

    Parameters
    ----------
    module_path : str
        e.g. "a.b.c.ClassName"

    Returns
    -------
    Tuple[str, str]
        e.g. ("a.b.c", "ClassName")
    """
    *m_path, cls = module_path.split(".")
    m_path = ".".join(m_path)
    return m_path, cls


def get_callable_kwargs(config: Dict, default_module: Union[str, ModuleType] = None) -> (type, Dict):
    """
    extract class/func and kwargs from config info

    Parameters
    ----------
    config : [dict, str]
        similar to config
        please refer to the doc of init_instance_by_config

    default_module : Python module or str
        It should be a python module to load the class type
        This function will load class from the config['module_path'] first.
        If config['module_path'] doesn't exists, it will load the class from default_module.

    Returns
    -------
    (type, dict):
        the class/func object and it's arguments.

    Raises
    ------
        ModuleNotFoundError
    """
    if isinstance(config, dict):
        key = "class" if "class" in config else "func"
        if isinstance(config[key], str):
            # 1) get module and class
            # - case 1): "a.b.c.ClassName"
            # - case 2): {"class": "ClassName", "module_path": "a.b.c"}
            m_path, cls = split_module_path(config[key])
            if m_path == "":
                m_path = config.get("module_path", default_module)
            module = get_module_by_module_path(m_path)

            # 2) get callable
            _callable = getattr(module, cls)  # may raise AttributeError
        else:
            _callable = config[key]  # the class type itself is passed in
        kwargs = config.get("kwargs", {})
    elif isinstance(config, str):
        # a.b.c.ClassName
        m_path, cls = split_module_path(config)
        module = get_module_by_module_path(default_module if m_path == "" else m_path)

        _callable = getattr(module, cls)
        kwargs = {}
    else:
        raise NotImplementedError(f"This type of input is not supported")
    return _callable, kwargs