# Alpha Mining - 深度强化学习的因子公式搜索

此项目为本人在2025 4-7月在私募基金实习所做的部分工作。
是一个基于强化学习的量化因子挖掘系统，支持多种**输入完全不同的公式表示**的神经网络架构的动态配置和训练。包含多种原创的实现。
项目仍在调试，在进行让模型通过假数据猜测预设公式的实验，随个人意愿推进

## 项目简介

Alpha Mining是一个创新的量化因子挖掘框架，通过深度强化学习自动发现和优化金融因子公式。
系统支持5种不同的神经网络架构，每种都有其独特的优势和应用场景。
为适应不同模型的不同输入格式，还实现了Formala类，能在不同类型间转换，满足所有模型的需求
用了比RPN（反波兰记号）更强的公式表示，几乎所有操作都能有时间偏移和时间窗口，避免摊平表示

### 主要特点
- 🚀 **5种原创模型**: 从基础LSTM到先进的因果Transformer
- 🔧 **动态配置**: JSON配置文件支持灵活的参数调整
- 🎯 **强化学习**: 基于PPO算法的智能体训练
- 📊 **量化应用**: 专门为金融因子挖掘设计
- 🛠️ **易于扩展**: 模块化设计支持新模型开发

## 目录

- [系统特性](#系统特性)
- [原创模型](#原创模型)
  - [LSTM_ag (基础LSTM模型)](#1-lstm_ag-基础lstm模型)
  - [LSTM_p_ag (双向LSTM模型)](#2-lstm_p_ag-双向lstm模型)
  - [LSTM_sa_ag (自注意力LSTM模型)](#3-lstm_sa_ag-自注意力lstm模型)
  - [ENCO_ca_ag (因果Transformer模型)](#4-enco_ca_ag-因果transformer模型)
  - [LSTM_ps_ag (Padded Sequence LSTM模型)](#5-lstm_ps_ag-padded-sequence-lstm模型)
- [安装和运行](#安装和运行)
- [配置文件结构](#配置文件结构)
- [模型选择建议](#模型选择建议)
- [性能调优](#性能调优)
- [故障排除](#故障排除)
- [技术架构总结](#技术架构总结)
- [扩展开发](#扩展开发)

## 系统特性

- **动态配置**: 通过JSON配置文件动态加载模型、智能体和环境
- **命令行驱动**: 支持命令行参数和配置覆盖
- **模块化设计**: 易于扩展新的模型和组件
- **多种公式支持**: 支持不同公式格式，适应不同模型

## 原创模型

### 1. LSTM_ag (基础LSTM模型)
**配置文件**: `configs/config_lstm.json`

**运行原理**:
LSTM_ag是一个基于递归LSTM的公式生成模型，采用树形结构处理数学公式。其核心思想是通过递归遍历公式树，为每个节点生成上下文表示。

**核心组件**:
- **嵌入层**: 
  - `factors_embedding_l/s`: 因子长期/短期记忆嵌入，存储对每个因子的记忆
  - `fn_embedding`: 函数类型嵌入，编码不同数学函数的语义
- **LSTM单元**: 多层LSTM处理序列信息，维护短期和长期记忆
- **注意力机制**: `AttentionNet`用于选择输入位置，增强模型的决策能力
- **函数参数融合器**: `fn_profile_fuser`将函数类型和时间参数融合

**工作流程**:
1. **递归遍历**: 从公式树根节点开始，递归访问每个子节点
2. **状态计算**: 对于每个节点，结合函数类型、时间参数和子节点状态
3. **记忆融合**: 将左右子节点的短期/长期记忆进行平均融合
4. **LSTM更新**: 使用当前节点信息更新LSTM状态
5. **动作生成**: 基于最终状态生成函数类型、参数和输入位置的概率分布
6. **注意力选择**: 使用`AttentionNet`选择输入位置，增强决策能力

**创新点**: 
- 首次将LSTM应用于数学公式生成，通过递归结构处理树形数据
- 引入注意力机制增强输入位置选择
- 支持因子记忆的长期和短期存储

**快速开始**: `python main.py --config configs/config_lstm.json --set training/num_epoch 2 training/num_episode_per_epoch 2 training/num_steps_per_episode 50`

### 2. LSTM_p_ag (双向LSTM模型)
**配置文件**: `configs/config_lstm_p.json`

**运行原理**:
LSTM_p_ag在LSTM_ag基础上引入了双向LSTM架构，通过同时考虑前向和后向信息来增强模型的上下文理解能力。

**核心组件**:
- **基础组件**: 继承LSTM_ag的所有嵌入层和LSTM单元
- **双向LSTM汇总器**: `bilstm_summerizer`处理序列的双向信息
- **并行处理**: 同时处理多个子节点的状态信息
- **状态拼接**: 将左右子节点状态在序列维度拼接后处理

**工作流程**:
1. **递归遍历**: 与LSTM_ag相同的递归遍历过程
2. **并行状态计算**: 同时计算左右子节点的状态，而不是串行处理
3. **状态拼接**: 将左右子节点的状态在序列维度上拼接
4. **LSTM处理**: 使用基础LSTM处理当前节点信息
5. **双向LSTM汇总**: 使用双向LSTM处理拼接后的状态序列
6. **最终汇总**: 通过双向LSTM的最终输出生成动作概率

**创新点**: 
- 引入双向LSTM增强上下文理解
- 并行处理子节点状态，提高效率
- 更好的长距离依赖建模能力
- 状态拼接机制增强信息融合

**特征**:
- 在基础LSTM基础上增加双向LSTM
- 更好的上下文理解能力
- 支持更复杂的序列模式识别

**快速开始**: `python main.py --config configs/config_lstm_p.json --set training/num_epoch 2 training/num_episode_per_epoch 2 training/num_steps_per_episode 50`

### 3. LSTM_sa_ag (自注意力LSTM模型)
**配置文件**: `configs/config_sa_ag.json`

**运行原理**:
LSTM_sa_ag是LSTM和Transformer的混合架构，通过自注意力机制和树形位置编码来增强模型的序列理解能力。

**核心组件**:
- **嵌入层**: 
  - `factors_embedding`: 因子嵌入
  - `fn_embedding`: 函数类型嵌入
  - `tree_pos_embedding`: 树形位置编码，编码节点在树中的位置
- **Transformer编码器**: `transformer_encoder`使用自注意力机制处理序列
- **双向LSTM**: `bilstm_summerizer`进一步处理Transformer的输出
- **参数融合器**: `profile_fuser`将函数类型和时间参数融合

**工作流程**:
1. **序列构建**: 通过`_proceed`方法递归构建序列，每个元素包含树位置和函数信息
2. **位置编码**: 为每个节点添加树形位置编码，表示其在公式树中的位置
3. **参数融合**: 将函数类型和时间参数融合为统一表示
4. **Transformer编码**: 使用自注意力机制处理整个序列，捕获全局依赖关系
5. **LSTM汇总**: 通过双向LSTM进一步处理Transformer的输出
6. **动作生成**: 基于最终表示生成动作概率分布

**创新点**:
- 首次将Transformer自注意力引入公式生成
- 树形位置编码保持结构信息
- 混合架构结合了Transformer的全局建模和LSTM的序列建模优势
- 递归序列构建机制处理树形结构

**特征**:
- 结合LSTM和Transformer架构
- 使用自注意力机制增强序列理解
- 支持树形位置编码
- 最先进的序列建模能力

**快速开始**: `python main.py --config configs/config_sa_ag.json --set training/num_epoch 2 training/num_episode_per_epoch 2 training/num_steps_per_episode 50`

### 4. ENCO_ca_ag (因果Transformer模型)
**配置文件**: `configs/config_enco.json`

**运行原理**:
ENCO_ca_ag是一个基于因果Transformer的模型，专门设计用于处理具有因果关系的序列数据，通过多层Transformer编码器逐步处理不同长度的序列。

**核心组件**:
- **嵌入层**: 
  - `embedding`: 统一的因子和函数嵌入
  - `positional_ebd`: 位置编码
  - `time_para_fc`: 时间参数编码
- **多层Transformer**: `transformer_encoders`数组，每层处理不同长度的序列
- **自动掩码机制**: `AutoMaskTransformerEncoder`实现因果注意力
- **因果掩码**: 上三角掩码确保位置t只能看到位置n<t的信息

**工作流程**:
1. **输入编码**: 结合位置编码、函数类型嵌入和时间参数
2. **分层处理**: 通过多个Transformer编码器，每层处理不同长度的序列
3. **因果掩码**: 使用上三角掩码确保位置t只能看到位置n<t的信息
4. **自动掩码**: 结合输入mask和因果mask，处理变长序列
5. **残差连接**: 每层输出与输入相加，保持信息流动
6. **动作生成**: 基于最终表示生成动作概率分布

**创新点**:
- 多层Transformer架构，每层专注不同序列长度
- 因果注意力机制确保时序一致性
- 自动掩码机制处理变长序列
- 高效的并行计算架构
- 统一嵌入层处理因子和函数

**特征**:
- 基于Transformer的因果编码器
- 支持自动掩码机制
- 适合处理具有因果关系的序列数据
- 高效的并行计算

**快速开始**: `python main.py --config configs/config_enco.json --set training/num_epoch 2 training/num_episode_per_epoch 2 training/num_steps_per_episode 50`

### 5. LSTM_ps_ag (Padded Sequence LSTM模型)
**配置文件**: `configs/config_lstm_ps.json`

**运行原理**:
LSTM_ps_ag是一个专门处理padded sequences的LSTM模型，通过结合Transformer和LSTM来处理变长序列，特别适合处理具有不同长度的公式序列。

**核心组件**:
- **嵌入层**: 
  - `fn_type_ebd`: 函数类型嵌入
  - `tree_pos_ebd`: 树位置嵌入
  - `time_para_fc`: 时间参数编码
- **LSTM层**: 处理padded sequences的主干网络
- **Transformer编码器**: 处理因子和公式的全局关系
- **汇总LSTM**: `summarizer_lstm`进一步处理序列信息
- **序列打包**: 使用`pack_padded_sequence`和`pad_packed_sequence`高效处理变长序列

**工作流程**:
1. **序列构建**: 构建包含树位置、函数类型和时间参数的序列
2. **序列打包**: 使用`pack_padded_sequence`处理变长序列，提高计算效率
3. **LSTM处理**: 通过LSTM处理打包后的序列，自动忽略padding部分
4. **序列解包**: 使用`pad_packed_sequence`恢复原始序列格式
5. **因子融合**: 将因子嵌入与公式表示拼接
6. **Transformer编码**: 使用Transformer处理全局依赖关系
7. **LSTM汇总**: 通过汇总LSTM生成最终表示
8. **动作生成**: 基于最终表示生成动作概率分布

**创新点**:
- 首次将padded sequence处理引入公式生成
- 结合LSTM和Transformer的混合架构
- 高效的变长序列处理机制
- 改进的训练稳定性和收敛性能
- 多层LSTM架构增强序列建模能力

**特征**:
- 支持padded sequences处理变长序列
- 结合LSTM和Transformer架构
- 改进的训练稳定性
- 更好的收敛性能

**快速开始**: `python main.py --config configs/config_lstm_ps.json --set training/num_epoch 2 training/num_episode_per_epoch 2 training/num_steps_per_episode 50`

## 快速开始

```bash
# 1. 安装
git clone https://github.com/LinChengHao3606307/AlphaMining.git

cd AlphaMining

conda env create -f environment.yml

conda activate LCH_Intern_Project_Env


# 2. 运行基础模型（快速测试）
python main.py --config configs/config_lstm.json --set training/num_epoch 2 training/num_episode_per_epoch 2 training/num_steps_per_episode 50

# 3. 运行其他模型
python main.py --config configs/config_sa_ag.json  # 自注意力LSTM
python main.py --config configs/config_enco.json   # 因果Transformer
python main.py --config configs/config_lstm_p.json # 双向LSTM
python main.py --config configs/config_lstm_ps.json # Padded Sequence LSTM
```

## 安装和运行

### 环境要求
- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- 其他依赖见requirements.txt

### 安装依赖
```bash
pip install torch numpy pandas
# 或使用requirements.txt
pip install -r requirements.txt
```

### 基本用法

1. **列出可用配置**:
```bash
python main.py --list-configs
```

2. **使用默认配置运行**:
```bash
python main.py --config configs/config_lstm.json
```

3. **使用特定模型配置**:
```bash
 # 运行LSTM_ag模型（基础模型）
python main.py --config configs/config_lstm.json

 # 运行LSTM_p_ag模型（双向LSTM）
python main.py --config configs/config_lstm_p.json

 # 运行LSTM_sa_ag模型（自注意力LSTM）
python main.py --config configs/config_sa_ag.json

 # 运行ENCO_ca_ag模型（因果Transformer）
python main.py --config configs/config_enco.json

 # 运行LSTM_ps_ag模型（Padded Sequence LSTM）
python main.py --config configs/config_lstm_ps.json
```

### 高级用法

1. **覆盖配置参数**:
```bash
 # 修改训练轮数
python main.py --config configs/config_sa_ag.json --set training/num_epoch 50

 # 修改学习率
python main.py --config configs/config_sa_ag.json --set agent/agent_class/kwargs/lr 0.001

 # 修改模型参数
python main.py --config configs/config_sa_ag.json --set upper_policy/kwargs/lstm_hidden_and_econ_dim 256

 # 修改ENCO模型参数
python main.py --config configs/config_enco.json --set upper_policy/kwargs/d_model 256

 # 修改LSTM_p模型参数
python main.py --config configs/config_lstm_p.json --set upper_policy/kwargs/lstm_hidden_size 256

 # 修改LSTM_ps模型参数
python main.py --config configs/config_lstm_ps.json --set upper_policy/kwargs/lstm_hidden_size 256
```

2. **多个参数覆盖**:
```bash
python main.py --config configs/config_sa_ag.json \
    --set training/num_epoch 50 \
    --set training/batch_size 8 \
    --set agent/agent_class/kwargs/lr 0.0005

 # 快速测试配置
python main.py --config configs/config_lstm.json \
    --set training/num_epoch 2 \
    --set training/num_episode_per_epoch 2 \
    --set training/num_steps_per_episode 50 \
    --set training/batch_size 2
```

## 配置文件结构

每个配置文件包含以下主要部分:

```json
{
  "random_seed": 42,
  "upper_policy": {
    "class": "模型类名",
    "module_path": "模块路径",
    "kwargs": {
      // 模型参数
    }
  },
  "upper_critic": {
    // 价值网络配置
  },
  "agent": {
    "agent_class": {
      "class": "智能体类名",
      "module_path": "模块路径",
      "kwargs": {
        // 智能体参数
      }
    }
  },
  "data": {
    "length": 1400,
    "start_date": [2018, 6, 1],
    "num_stocks": 2
  },
  "target_formula": [
    // 目标公式列表
  ],
  "environment": {
    // 环境配置
  },
  "training": {
    "num_epoch": 100,
    "num_episode_per_epoch": 5,
    "num_steps_per_episode": 500,
    "batch_size": 4
  }
}
```

### 主要配置参数说明
- **random_seed**: 随机种子，确保实验可重现
- **upper_policy**: 策略网络配置，包含模型类和参数
- **upper_critic**: 价值网络配置，用于评估状态价值
- **agent**: 强化学习智能体配置，支持PPO等算法
- **data**: 数据配置，包括数据长度、起始日期和股票数量
- **target_formula**: 目标公式列表，用于训练和评估
- **environment**: 环境配置，包括公式长度、训练参数等
- **training**: 训练参数，包括轮数、批次大小等

## 模型选择建议

### 模型对比表

| 模型 | 架构特点 | 适用场景 | 计算复杂度 | 效果 |
|------|----------|----------|------------|------------|
| **LSTM_ag** | 基础递归LSTM + 注意力 | 入门学习、简单任务 | 高 | 差 |
| **LSTM_p_ag** | 双向LSTM + 并行处理 | 需要上下文理解 | 高 | 中 |
| **LSTM_sa_ag** | Transformer + LSTM混合 | 复杂任务、长距离依赖 | 中 | 中 |
| **ENCO_ca_ag** | 因果Transformer | 因果推理、时序关系 | 中 | 中 |
| **LSTM_ps_ag** | Padded Sequence + LSTM | 变长序列处理 | 中 | 良 |

### 详细建议

- **LSTM_ag**: 基础模型，适合入门学习和简单任务，计算效率高，训练稳定，支持注意力机制
- **LSTM_p_ag**: 需要更好上下文理解时使用，双向LSTM提供更丰富的序列信息，并行处理提高效率
- **LSTM_sa_ag**: 最先进的性能，结合Transformer自注意力，适合复杂任务和长距离依赖，支持树形位置编码
- **ENCO_ca_ag**: 适合需要因果推理的任务，多层Transformer架构处理时序关系，支持自动掩码机制
- **LSTM_ps_ag**: 适合处理变长序列，padded sequence机制提高训练效率，多层LSTM增强建模能力

## 性能调优

1. **学习率调整**: 使用`--set agent/agent_class/kwargs/lr`调整
2. **批次大小**: 使用`--set training/batch_size`调整
3. **模型容量**: 调整隐藏层维度
   - LSTM_ag/LSTM_p_ag: `--set upper_policy/kwargs/lstm_hidden_size`
   - LSTM_sa_ag: `--set upper_policy/kwargs/lstm_hidden_and_econ_dim`
   - ENCO_ca_ag: `--set upper_policy/kwargs/d_model`
   - LSTM_ps_ag: `--set upper_policy/kwargs/lstm_hidden_size`
4. **训练轮数**: 根据收敛情况调整epoch数
5. **LSTM层数**: 调整`--set upper_policy/kwargs/lstm_num_layers`
6. **Transformer层数**: 调整`--set upper_policy/kwargs/enco_num_layers`或`--set upper_policy/kwargs/num_layers`

## 故障排除

1. **内存不足**: 减小batch_size或模型维度
2. **收敛慢**: 调整学习率或增加训练轮数
3. **过拟合**: 增加dropout或减少模型容量
4. **梯度爆炸**: 使用梯度裁剪或调整学习率
5. **训练不稳定**: 检查数据预处理和模型初始化
6. **序列长度问题**: 调整formula_profile_length参数

## 常见问题

### Q: 如何选择最适合的模型？
A: 参考[模型选择建议](#模型选择建议)部分，根据你的具体需求选择合适的模型。

### Q: 训练时间太长怎么办？
A: 可以减小`num_epoch`、`num_episode_per_epoch`、`num_steps_per_episode`等参数，或使用更小的模型。

### Q: 如何添加新的神经网络模型？
A: 参考[扩展开发](#扩展开发)部分，按照接口要求实现新的模型类。

### Q: 支持哪些数据类型？
A: 目前支持Excel格式的股票数据，可以扩展支持其他格式。

### Q: 如何调整模型性能？
A: 参考[性能调优](#性能调优)部分，调整学习率、批次大小、模型容量等参数。

## 技术架构总结

### 模型演进路线

```
LSTM_ag (基础) 
    ↓ 增加双向LSTM
LSTM_p_ag (双向)
    ↓ 增加Transformer
LSTM_sa_ag (混合)
    ↓ 纯Transformer
ENCO_ca_ag (因果)
    ↓ 增加序列打包
LSTM_ps_ag (变长)
```

### 技术演进细节
1. **LSTM_ag**: 基础递归LSTM → 树形结构处理 + 注意力机制
2. **LSTM_p_ag**: 双向LSTM → 增强上下文理解 + 并行处理
3. **LSTM_sa_ag**: Transformer自注意力 → 全局依赖建模 + 树形位置编码
4. **ENCO_ca_ag**: 因果Transformer → 时序一致性 + 自动掩码
5. **LSTM_ps_ag**: Padded Sequence → 变长序列处理 + 多层LSTM

### 核心技术创新
- **递归树遍历**: 将数学公式表示为树结构进行递归处理，完全符合公式运行的顺序
- **混合架构**: 结合LSTM和Transformer的优势
- **位置编码**: 树形位置编码保持结构信息
- **因果注意力**: 确保时序一致性和因果关系
- **序列打包**: 高效处理变长序列
- **注意力机制**: 增强输入位置选择和决策能力
- **因子记忆**: 长期和短期记忆存储机制
- **自动掩码**: 处理变长序列和因果约束

### 应用场景
- **量化因子挖掘**: 自动发现有效的金融因子和投资策略
- **数学公式生成**: 生成复杂的数学表达式和函数组合

## 扩展开发

要添加新模型:
1. 在`main/Agent/models/`中创建模型类，继承`nn.Module`
2. 对于输入格式不符合现有格式的模型，实现Fromula类的新格式
3. 实现`forward(input_state)`方法，返回动作概率分布和参数
4. 在`configs/`中创建配置文件，指定模型类和参数
5. 确保模型类有正确的接口和权重初始化
6. 可选：实现`save(name)`和`load(name)`方法用于模型保存和加载

### 模型接口要求
```python
def forward(self, input_state: torch.Tensor):
    """
    输入: input_state - 形状为[batch_size, formula_profile_length, ...]的张量
    输出: (discrete_probs, (mean, std)) - 离散动作概率和连续参数
    """
    pass
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
