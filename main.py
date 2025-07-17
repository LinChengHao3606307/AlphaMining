"""
    动态初始化训练系统
    通过配置文件来指定模型类和参数，而不是硬编码import
"""
import time
import json
import importlib
import numpy as np
import torch
from __init__ import DefaultValues, Formula
import pandas as pd
from datetime import timedelta
from main.Env.DataSet.data_sets.stimulate_data import generate_dataframe


def get_sep_idx(df: pd.DataFrame, dates: list[str] = ["20211231", "20221231"]) -> tuple[int, int]:
    result: list[int] = []
    oneday = timedelta(days=1)
    for date in dates:
        date_pd = pd.to_datetime(date, format='%Y%m%d')
        seps = df.index[df['date'] == date_pd]
        while len(seps) == 0:
            date_pd -= oneday
            seps = df.index[df['date'] == date_pd]
        result.append(seps[-1].item())
    return (result[0], result[1])


def dynamic_import_and_instantiate(config: dict):
    """
    动态导入并实例化类
    
    Args:
        config: 包含class, module_path, kwargs的字典
    
    Returns:
        实例化的对象
    """
    try:
        # 动态导入模块
        module = importlib.import_module(config["module_path"])
        # 获取类
        class_obj = getattr(module, config["class"])
        
        # 处理嵌套的kwargs
        kwargs = config.get("kwargs", {})
        processed_kwargs = {}
        
        for key, value in kwargs.items():
            if isinstance(value, dict) and "class" in value and "module_path" in value:
                # 这是一个嵌套的配置对象，需要递归实例化
                processed_kwargs[key] = dynamic_import_and_instantiate(value)
            else:
                processed_kwargs[key] = value
        
        # 实例化
        instance = class_obj(**processed_kwargs)
        return instance
    except Exception as e:
        print(f"动态导入失败: {e}")
        raise


def train(conf_dict: dict):
    """
    使用配置字典进行训练
    
    Args:
        conf_dict: 包含所有训练配置的字典
    """
    print("开始动态初始化训练...")
    print(f"配置: {json.dumps(conf_dict, indent=2, ensure_ascii=False)}")
    
    # 设置随机种子
    if conf_dict.get("random_seed") is not None:
        torch.manual_seed(conf_dict["random_seed"])
        np.random.seed(conf_dict["random_seed"])
    
    # 动态初始化上游策略网络
    print("初始化上游策略网络...")
    upper_m_config = conf_dict["upper_policy"]
    upper_m = dynamic_import_and_instantiate(upper_m_config)
    upper_m = upper_m.to(DefaultValues.device)
    
    # 动态初始化价值网络
    print("初始化价值网络...")
    upper_crit_config = conf_dict["upper_critic"]
    upper_crit = dynamic_import_and_instantiate(upper_crit_config)
    upper_crit = upper_crit.to(DefaultValues.device)
    
    # 初始化PPO代理
    print("初始化PPO代理...")
    from main.Agent import HybridPPO
    agent_config = conf_dict["agent"]
    agent = HybridPPO(
        policy=upper_m,
        value_net=upper_crit,
        state_type=agent_config.get("state_type", "tpv"),
        lr=agent_config.get("lr", 1e-3),
        buffer_size=agent_config.get("buffer_size", 500),
        gamma=agent_config.get("gamma", 1-1/DefaultValues.formula_profile_length),
        clip_eps=agent_config.get("clip_eps", 0.5),
        clip_method=agent_config.get("clip_method", "epoch"),
        ent_coef=agent_config.get("ent_coef", 5e-2)
    )
    
    # 动态初始化下游环境
    print("初始化下游环境...")
    from main.Env import Env
    from main.Env.models.Linear_ac import Linear_ac
    import torch.nn as nn
    
    # 生成目标公式
    tgt_formula = conf_dict.get("target_formula", [
        (9, -2, 0, 1, -6, 0, 1),
        (9, 0, 0, 1, -9, 0, 1),
        (9, 1, 0, 1, 0, 0, 1),
        (1, -1, 0, 1, 0, 0, 1),
        (3, 3, 0, 1, -2, 0, 1),
    ])
    
    # 生成数据
    data_config = conf_dict.get("data", {})
    dfs = [generate_dataframe(
        data_config.get("length", 1400),
        data_config.get("start_date", (2018, 6, 1)),
        tgt_formula
    ) for _ in range(data_config.get("num_stocks", 2))]
    
    print(f"生成数据: {len(dfs)} 只股票，每只 {len(dfs[0])} 个时间点")
    
    # 获取分割点
    seps = get_sep_idx(dfs[0])
    
    # 初始化环境
    env_config = conf_dict.get("environment", {})
    env = Env(
        original_data=dfs,
        split_idx=seps,
        lower_model_class=Linear_ac,
        criterion=nn.MSELoss(),
        num_formulas=env_config.get("num_formulas", DefaultValues.total_amount_of_formulas),
        formula_profile_length=env_config.get("formula_profile_length", DefaultValues.formula_profile_length),
        lower_train_epoch=env_config.get("lower_train_epoch", 1),
        lower_train_batch=env_config.get("lower_train_batch", 256),
        lower_train_lr=env_config.get("lower_train_lr", 0.001),
        bound=env_config.get("bound", (-1, 4)),
        sign_division=env_config.get("sign_division", 0.005)
    )
    
    # 训练循环
    print("开始训练循环...")
    training_config = conf_dict.get("training", {})
    num_epoch = training_config.get("num_epoch", 100)
    num_episode_per_epoch = training_config.get("num_episode_per_epoch", 5)
    num_steps_per_episode = training_config.get("num_steps_per_episode", 500)
    batch_size = training_config.get("batch_size", 4)
    
    best_loss = None
    
    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1}/{num_epoch}")
        env.tdh_r.start_new_epoch_record()
        env.tdh_l.start_new_epoch_record()
        
        for episode in range(num_episode_per_epoch):
            state = env.reset()
            buffer_full = False
            
            t = time.perf_counter()
            while not buffer_full:
                # 选择动作
                action_dict = agent.act(state)
                actt = {
                    'discrete': action_dict['discrete'],
                    'continuous': action_dict['continuous']
                }
                
                if not len(state) == DefaultValues.formula_profile_length:
                    f = Formula(std_formula=state.std_formula)
                    f.append_sfn_fn(actt)
                else:
                    f = Formula().get_dummy()
                
                next_state, reward, done, _ = env.step(f.std_formula[-1])
                if done:
                    f.show()
                    print(f"Reward: {reward}")
                
                # 存储转移
                buffer_full = agent.store_transition(state, action_dict, reward, done)
                state = next_state
            
            print("目标公式:")
            for fn in tgt_formula:
                for i in fn:
                    print(i, end="\t\t")
                print()
            print(f"Lower训练时间: {time.perf_counter() - t:.4f}s")
            
            t = time.perf_counter()
            agent.update(batch_size=batch_size)
            print(f"Upper训练时间: {time.perf_counter() - t:.4f}s")
        
        # 记录损失
        epoch_loss = env.tdh_r.get_last_epoch_avg_loss(category="reward")
        print(f"Epoch {epoch + 1} 平均奖励: {epoch_loss:.6f}")
        
        if best_loss is None:
            best_loss = epoch_loss
        elif epoch_loss < best_loss:
            agent.save("best")
            best_loss = epoch_loss
            print(f"新的最佳损失: {best_loss:.6f}")
        
        agent.save("last")
    
    print("训练完成!")


if __name__ == "__main__":
    # 默认配置
    default_config = {
        "random_seed": 42,
        "upper_policy": {
            "class": "LSTM_ag",
            "module_path": "main.Agent.models.LSTM_ag",
            "kwargs": {
                "total_amount_of_factors": DefaultValues.total_amount_of_factors,
                "total_types_of_fn": DefaultValues.total_types_of_fn,
                "formula_profile_length": DefaultValues.formula_profile_length,
                "lstm_hidden_size": 128,
                "lstm_num_layers": 3
            }
        },
        "upper_critic": {
            "class": "Value_est",
            "module_path": "main.Agent.models.Value_est",
            "kwargs": {
                "main_model": {
                    "class": "LSTM_ag",
                    "module_path": "main.Agent.models.LSTM_ag",
                    "kwargs": {
                        "total_amount_of_factors": DefaultValues.total_amount_of_factors,
                        "total_types_of_fn": DefaultValues.total_types_of_fn,
                        "formula_profile_length": DefaultValues.formula_profile_length,
                        "lstm_hidden_size": 128,
                        "lstm_num_layers": 3
                    }
                },
                "discrete_dims": None,
                "continuous_dim": 4,
                "mid_size": 2048
            }
        },
        "agent": {
            "state_type": "tpv",
            "lr": 1e-3,
            "buffer_size": 500,
            "gamma": 1-1/DefaultValues.formula_profile_length,
            "clip_eps": 0.5,
            "clip_method": "epoch",
            "ent_coef": 5e-2
        },
        "data": {
            "length": 1400,
            "start_date": (2018, 6, 1),
            "num_stocks": 2
        },
        "target_formula": [
            (9, -2, 0, 1, -6, 0, 1),
            (9, 0, 0, 1, -9, 0, 1),
            (9, 1, 0, 1, 0, 0, 1),
            (1, -1, 0, 1, 0, 0, 1),
            (3, 3, 0, 1, -2, 0, 1),
        ],
        "environment": {
            "num_formulas": DefaultValues.total_amount_of_formulas,
            "formula_profile_length": DefaultValues.formula_profile_length,
            "lower_train_epoch": 1,
            "lower_train_batch": 256,
            "lower_train_lr": 0.001,
            "bound": (-1, 4),
            "sign_division": 0.005
        },
        "training": {
            "num_epoch": 100,
            "num_episode_per_epoch": 5,
            "num_steps_per_episode": 500,
            "batch_size": 4
        }
    }
    
    # 从配置文件加载配置（如果存在）
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        print("从config.json加载配置")
    except FileNotFoundError:
        config = default_config
        print("使用默认配置")
        # 保存默认配置到文件
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("默认配置已保存到config.json")
    
    train(config) 