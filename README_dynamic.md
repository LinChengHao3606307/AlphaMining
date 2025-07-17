# 动态初始化训练系统

这个系统允许通过配置文件来动态指定模型类和参数，而不需要硬编码import语句。

## 文件结构

- `main.py`: 主要的动态初始化训练脚本
- `config.json`: 默认配置文件（首次运行时会自动生成）
- `config_examples.json`: 基础LSTM模型配置示例
- `config_lstm_p.json`: LSTM_p_ag模型配置示例
- `config_enco.json`: ENCO_ca_ag模型配置示例
- `README_dynamic.md`: 本说明文件

## 使用方法

### 1. 基本使用

```bash
python main.py
```

首次运行时会自动生成默认配置文件 `config.json`。

### 2. 使用自定义配置

将任意配置文件复制为 `config.json`，然后运行：

```bash
cp config_lstm_p.json config.json
python main.py
```

### 3. 配置文件格式

配置文件是一个JSON格式的字典，包含以下主要部分：

#### 上游策略网络配置
```json
{
  "upper_policy": {
    "class": "LSTM_ag",
    "module_path": "main.Agent.models.LSTM_ag",
    "kwargs": {
      "total_amount_of_factors": 31,
      "total_types_of_fn": 24,
      "formula_profile_length": 5,
      "lstm_hidden_size": 128,
      "lstm_num_layers": 3
    }
  }
}
```

#### 价值网络配置
```json
{
  "upper_critic": {
    "class": "Value_est",
    "module_path": "main.Agent.models.Value_est",
    "kwargs": {
      "main_model": {
        "class": "LSTM_ag",
        "module_path": "main.Agent.models.LSTM_ag",
        "kwargs": {
          "total_amount_of_factors": 31,
          "total_types_of_fn": 24,
          "formula_profile_length": 5,
          "lstm_hidden_size": 128,
          "lstm_num_layers": 3
        }
      },
      "discrete_dims": null,
      "continuous_dim": 4,
      "mid_size": 2048
    }
  }
}
```

#### PPO代理配置
```json
{
  "agent": {
    "state_type": "tpv",
    "lr": 1e-3,
    "buffer_size": 500,
    "gamma": 0.8,
    "clip_eps": 0.5,
    "clip_method": "epoch",
    "ent_coef": 5e-2
  }
}
```

#### 数据配置
```json
{
  "data": {
    "length": 1400,
    "start_date": [2018, 6, 1],
    "num_stocks": 2
  }
}
```

#### 环境配置
```json
{
  "environment": {
    "num_formulas": 1,
    "formula_profile_length": 5,
    "lower_train_epoch": 1,
    "lower_train_batch": 256,
    "lower_train_lr": 0.001,
    "bound": [-1, 4],
    "sign_division": 0.005
  }
}
```

#### 训练配置
```json
{
  "training": {
    "num_epoch": 100,
    "num_episode_per_epoch": 5,
    "num_steps_per_episode": 500,
    "batch_size": 4
  }
}
```

## 可用的模型

### 1. LSTM_ag
- **模块路径**: `main.Agent.models.LSTM_ag`
- **状态类型**: `tpv`
- **特点**: 基础LSTM模型，使用树位置向量状态

### 2. LSTM_p_ag
- **模块路径**: `main.Agent.models.LSTM_p_ag`
- **状态类型**: `buf`
- **特点**: 使用buffer格式状态的LSTM模型

### 3. ENCO_ca_ag
- **模块路径**: `main.Agent.models.ENCO_ca_ag`
- **状态类型**: `tpv`
- **特点**: 基于Transformer的编码器模型

### 4. LSTM_sa_ag
- **模块路径**: `main.Agent.models.LSTM_sa_ag`
- **状态类型**: `tpv`
- **特点**: 带自注意力的LSTM模型

### 5. LSTM_ps_ag
- **模块路径**: `main.Agent.models.LSTM_ps_ag`
- **状态类型**: `pse`
- **特点**: 结合LSTM和Transformer的混合模型

## 价值网络选择

### 1. Value_est
- **模块路径**: `main.Agent.models.Value_est`
- **特点**: 通用价值估计网络，需要指定main_model

### 2. LSTM_crit
- **模块路径**: `main.Agent.models.LSTM_ag`
- **特点**: 基于LSTM的评论家网络

### 3. LSTM_p_crit
- **模块路径**: `main.Agent.models.LSTM_p_ag` 或 `main.Agent.models.ENCO_ca_ag`
- **特点**: 基于LSTM_p的价值网络

## 配置示例

### 使用LSTM_ag模型
```bash
cp config_examples.json config.json
python main.py
```

### 使用LSTM_p_ag模型
```bash
cp config_lstm_p.json config.json
python main.py
```

### 使用ENCO_ca_ag模型
```bash
cp config_enco.json config.json
python main.py
```

## 高级配置

### 嵌套对象配置
系统支持嵌套的对象配置，例如在Value_est中指定main_model：

```json
{
  "upper_critic": {
    "class": "Value_est",
    "module_path": "main.Agent.models.Value_est",
    "kwargs": {
      "main_model": {
        "class": "LSTM_ag",
        "module_path": "main.Agent.models.LSTM_ag",
        "kwargs": {
          "lstm_hidden_size": 128,
          "lstm_num_layers": 3
        }
      }
    }
  }
}
```

### 自定义目标公式
```json
{
  "target_formula": [
    [9, -2, 0, 1, -6, 0, 1],
    [9, 0, 0, 1, -9, 0, 1],
    [9, 1, 0, 1, 0, 0, 1],
    [1, -1, 0, 1, 0, 0, 1],
    [3, 3, 0, 1, -2, 0, 1]
  ]
}
```

## 注意事项

1. **模块路径**: 确保模块路径正确，相对于项目根目录
2. **类名**: 确保类名在指定模块中存在
3. **参数**: 确保kwargs中的参数与类的__init__方法匹配
4. **状态类型**: 不同模型可能需要不同的状态类型（tpv, buf, pse等）
5. **设备**: 模型会自动移动到DefaultValues.device指定的设备上

## 错误处理

如果动态导入失败，系统会打印详细的错误信息，包括：
- 模块导入错误
- 类不存在错误
- 参数不匹配错误

请检查配置文件中的模块路径、类名和参数是否正确。 