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


def apply_config_overrides(config: dict, overrides: list) -> dict:
    """
    应用配置覆盖
    
    Args:
        config: 原始配置字典
        overrides: 覆盖参数列表，格式为 [path1, value1, path2, value2, ...]
    
    Returns:
        修改后的配置字典
    """
    if len(overrides) % 2 != 0:
        print("错误: --set 参数必须是成对的路径和值")
        exit(1)
    
    # 创建配置的深拷贝
    import copy
    config = copy.deepcopy(config)
    
    for i in range(0, len(overrides), 2):
        path = overrides[i]
        value = overrides[i + 1]
        
        # 解析路径
        path_parts = path.split('/')
        current = config
        
        # 导航到父节点
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # 设置值
        key = path_parts[-1]
        
        # 尝试转换值的类型
        try:
            # 尝试转换为数字
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # 如果不是数字，保持字符串
            pass
        
        # 特殊处理布尔值
        if isinstance(value, str) and value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        
        current[key] = value
        print(f"设置 {path} = {value}")
    
    return config


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
    
    # 动态初始化PPO代理
    print("初始化PPO代理...")
    agent_config = conf_dict["agent"]
    
    # 检查是否指定了自定义agent类
    if "agent_class" in agent_config:
        agent = dynamic_import_and_instantiate(agent_config["agent_class"])
        # 设置策略和价值网络
        agent.set_policy(upper_m)
        agent.set_value_net(upper_crit)
    else:
        # 使用默认的HybridPPO
        from main.Agent.agents.hybridPPO import HybridPPO
        agent = HybridPPO(
            policy=upper_m,
            value_net=upper_crit,
            state_type=agent_config.get("state_type", "buf"),
            lr=agent_config.get("lr", 1e-3),
            buffer_size=agent_config.get("buffer_size", 500),
            gamma=agent_config.get("gamma", 1-1/DefaultValues.formula_profile_length),
            clip_eps=agent_config.get("clip_eps", 0.5),
            clip_method=agent_config.get("clip_method", "epoch"),
            ent_coef=agent_config.get("ent_coef", 5e-2)
        )
    
    # 动态初始化下游环境
    print("初始化下游环境...")
    
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
    
    # 检查是否指定了自定义环境类
    env_config = conf_dict.get("environment", {})
    if "environment_class" in env_config:
        # 动态初始化自定义环境
        env = dynamic_import_and_instantiate(env_config["environment_class"])
        # 设置环境参数
        env.setup(
            original_data=dfs,
            split_idx=seps,
            target_formula=tgt_formula,
            **env_config.get("kwargs", {})
        )
    else:
        # 使用默认的Env
        from main.Env import Env
        from main.Env.models.Linear_ac import Linear_ac
        import torch.nn as nn
        
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
    import argparse
    import os
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="动态初始化训练系统")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config_lstm.json",
        help="配置文件路径 (默认: configs/config_lstm.json)"
    )
    parser.add_argument(
        "--list-configs", 
        action="store_true",
        help="列出所有可用的配置文件"
    )
    parser.add_argument(
        "--set",
        nargs="+",
        metavar="PATH VALUE",
        help="设置配置参数，格式: path value [path value ...]"
    )
    
    args = parser.parse_args()
    
    # 如果用户要求列出配置文件
    if args.list_configs:
        config_dir = "configs"
        if os.path.exists(config_dir):
            config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
            print("可用的配置文件:")
            for config_file in sorted(config_files):
                print(f"  - {config_file}")
        else:
            print("configs目录不存在")
        exit(0)
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 '{args.config}' 不存在")
        print("使用 --list-configs 查看可用的配置文件")
        exit(1)
    
    # 从配置文件加载配置
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"从 {args.config} 加载配置")
    except Exception as e:
        print(f"错误: 无法加载配置文件 '{args.config}': {e}")
        exit(1)
    
    # 处理命令行参数覆盖
    if args.set:
        config = apply_config_overrides(config, args.set)
    
    train(config) 