
"""
    训练loop，把所有代码汇总于此进行训练
"""
import time

import numpy as np
import torch

from others.alpha_mining import DefaultValues, Formula
# 设置随机种子
# torch.manual_seed(0)
# np.random.seed(0)
import pandas as pd
from datetime import  timedelta

from others.alpha_mining.main.Env.DataSet.data_sets.stimulate_data import generate_dataframe


def get_sep_idx(df:pd.DataFrame, dates:list[str]=["20211231","20221231"])->tuple[int,...]:
    result:list[int] = []
    oneday = timedelta(days=1)
    for date in dates:
        date_pd = pd.to_datetime(date, format='%Y%m%d')
        seps = df.index[df['date'] == date_pd]
        while len(seps) == 0:
            date_pd -= oneday
            seps = df.index[df['date'] == date_pd]
        result.append(seps[-1])
    return tuple(result)

def train(lr:float=1e-3,num_epoch:int=100,num_episode_per_epoch:int=5,num_steps_per_episode:int=500):
    # 上游布置
    from Agent import HybridPPO
    from Agent.models.LSTM_ag import LSTM_ag, LSTM_crit
    from Agent.models.LSTM_p_ag import LSTM_p_ag, LSTM_p_crit
    from Agent.models.LSTM_sa_ag import LSTM_sa_ag
    from Agent.models.LSTM_ps_ag import LSTM_ps_ag
    from Agent.models.Value_est import Value_est
    upper_m = LSTM_ps_ag().to(DefaultValues.device)
    upper_crit = Value_est(main_model=LSTM_ps_ag()).to(DefaultValues.device)
    agent = HybridPPO(policy=upper_m,value_net=upper_crit, state_type="pse", lr=lr,buffer_size=num_steps_per_episode)

    # 下游布置
    from Env import Env
    from Env.DataSet.data_sets.real_data import get_stock_data
    #dfs = get_stock_data(stock_idx_range=(0,50))
    tgt_formula = [
                                  (3,
                                   -2, -5, 1,
                                   -6, -6, 1),
                                  (5,
                                   0, -2, 1,
                                   -9, -1, 1),
                                  (19,
                                   1, -2, 1,
                                   0, 0, 1),
                                  (0,
                                   1, 0, 1,
                                   0, 0, 1),
                                  (8,
                                   3, 0, 1,
                                   2, -1, 1),
                              ]
    tgt_formula = [
                                  (3,
                                   -2, 0, 1,
                                   -6, 0, 1),
                                  (5,
                                   0, 0, 1,
                                   -9, 0, 1),
                                  (19,
                                   1, 0, 1,
                                   0, 0, 1),
                                  (0,
                                   1, 0, 1,
                                   0, 0, 1),
                                  (8,
                                   3, 0, 1,
                                   2, 0, 1),
                              ]
    tgt_formula = [
                                  (9,
                                   -2, 0, 1,
                                   -6, 0, 1),
                                  (9,
                                   0, 0, 1,
                                   -9, 0, 1),
                                  (9,
                                   1, 0, 1,
                                   0, 0, 1),
                                  (1,
                                   -1, 0, 1,
                                   0, 0, 1),
                                  (3,
                                   3, 0, 1,
                                   -2, 0, 1),
                              ]
    """
    tgt_formula = [
                                  (9,
                                   -1, 0, 1,
                                   -1, 0, 1),
                                  (9,
                                   0, 0, 1,
                                   -1, 0, 1),
                                  (9,
                                   1, 0, 1,
                                   -1, 0, 1),
                                  (9,
                                   2, 0, 1,
                                   -1, 0, 1),
                                  (9,
                                   3, 0, 1,
                                   -1, 0, 1),
                              ]
    """
    dfs = [generate_dataframe(1400,(2018,6,1),
                              tgt_formula
                              ) for _ in range(2)]
    print(dfs[0])
    # 21前训练，22验证，23不碰
    seps = get_sep_idx(dfs[0])
    from Env.models.Linear_ac import Linear_ac
    #lower_m = Linear_ac(ip_dim=DefaultValues.total_amount_of_factors+DefaultValues.total_amount_of_formulas+1)
    from Env.Criterion.customed_cri import CustomizedCri
    import torch.nn as nn

    env = Env(original_data=dfs,split_idx=seps,lower_model_class=Linear_ac,criterion=nn.MSELoss())

    # 训练循环
    best_loss = None
    agent.load("last")
    for epoch in range(num_epoch):
        env.tdh_r.start_new_epoch_record()
        env.tdh_l.start_new_epoch_record()
        for episode in range(num_episode_per_epoch):
            state, phrase_transition = env.reset()
            if phrase_transition:
                agent.policy.initialize_weights()
            buffer_full = False

            t = time.perf_counter()
            while not buffer_full:
                # 选择动作

                action_dict = agent.act(state)
                if not len(state) == DefaultValues.formula_profile_length:
                    f = Formula(std_formula=state.std_formula)
                    f.append_sfn_fn(action_dict)
                else:
                    f = Formula().get_dummy()


                next_state, reward, done, _ = env.step(f)
                if done:
                    f.show()
                    print(reward)

                # 存储转移
                buffer_full = agent.store_transition(state, action_dict, reward, done)

                state = next_state
            print("tgt " * 10)
            for fn in tgt_formula:
                for i in fn:
                    print(i, end="\t\t")
                print()
            print("lower:")
            print(time.perf_counter() - t)

            t = time.perf_counter()
            agent.update(batch_size=32)
            print("upper:")
            print(time.perf_counter()-t)

        epoco_loss = env.tdh_r.get_last_epoch_avg_loss(category="reward")
        if best_loss is None:
            best_loss = epoco_loss
        elif epoco_loss<best_loss:
            agent.save("best")
            best_loss = epoco_loss
        agent.save("last")

if __name__ == "__main__":
    # with torch.autograd.set_detect_anomaly(True):
    train()

