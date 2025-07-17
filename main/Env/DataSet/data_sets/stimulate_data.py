import pandas as pd
import numpy as np
from datetime import datetime, timedelta


from main.Env.DataSet import FactorComputer


from __init__ import DefaultValues


def generate_dataframe(t: int, start_date:tuple[int,int,int],formula,
                       f: int=DefaultValues.total_amount_of_factors, noise_scale:float=0.02) -> pd.DataFrame:
    """
    生成指定格式的DataFrame

    参数:
    t: int - 总天数/总索引数
    f: int - 总因子数

    返回:
    pd.DataFrame - 生成的DataFrame
    """
    # 生成日期列
    base_date = datetime(start_date[0], start_date[1], start_date[2])
    dates = [base_date + timedelta(days=i) for i in range(t)]
    date_strs = [date.strftime("%Y%m%d") for date in dates]

    # 生成return列 (随机数)
    returns = np.random.randn(t)*noise_scale

    # 生成因子列
    factors = {}
    for i in range(1, f + 1):
        factors[f"factor_{i}"] = np.random.randn(t)

    # 创建DataFrame
    df = pd.DataFrame({
        "date": date_strs,
        "return": returns
    })

    # 添加因子列
    for i, (name, values) in enumerate(factors.items(), start=2):
        df.insert(i, name, values)

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

    factor_computer = FactorComputer(df.iloc[:, 2:].to_numpy(), t, f)
    factor_computer.set_all_fn(formula)
    all_results = factor_computer.compute_vectorized()
    df["return"] += all_results
    return df


# 示例用法
if __name__ == "__main__":
    df = generate_dataframe(t=1000, f=5,start_date=(2020,2,1))
    print(df)
    print(df.index[df['date'] == "20200202"].item())