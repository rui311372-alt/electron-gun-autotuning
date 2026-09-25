# 电子枪光斑 FWHM 闭环优化系统

本项目是一个**电子枪光斑质量自动优化系统**：
- 采集电子枪光斑的 14-bit CCD 图像
- 分析图像计算光斑质量得分（score = sqrt(x1^2 + x2^2), 失败时为 nan，越小越好）
- 通过 Modbus 控制电子枪参数
- 实现闭环优化，自动调整参数以获得最佳光斑质量

## 两个核心入口

| 文件 | 定位 | 优化方式 | 适用场景 |
|------|------|----------|----------|
| **app.py** | 命令行总入口 | 手动参数序列 | 调试、单步操作、精确控制 |
| **bayesian_optimizer.py** | 贝叶斯优化器 | 自动参数搜索 | 自动寻优、参数空间探索 |

## 快速开始（已完成）

```powershell
# 创建虚拟环境
conda create -n egun_differential python=3.10
conda activate egun_differential

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 一、app.py（命令行总入口）

适用于：手动控制、调试、精确参数序列

# Modbus 控制
python app.py modbus-on    # 电源使能ON
python app.py modbus-set --Ua 90 --Ia 100 --Uc 660 --If 0.46 --Ug 200
# python app.py modbus-set --Ua 70 --Ia 100 --Uc 650 --If 0.42 --Ug 200
python app.py modbus-read  # 读取反馈
python app.py modbus-off   # 电源使能OFF

# 仅采集图像
python app.py capture

# 仅分析已有图像
python app.py analyze

### 二、bayesian_optimizer.py（贝叶斯优化器）

适用于：自动寻优、参数空间探索

```powershell
# 启动贝叶斯优化（自动搜索最优参数）
python bayesian_optimizer.py --max-iter 30 --init-points 5
```

**参数说明**：
- `--max-iter`: 最大迭代次数（默认30）
- `--init-points`: 初始随机采样点数（默认5）

**特点**：
- 基于 Gaussian Process 的贝叶斯优化
- 参数范围自动从 `gun_modbus_config.json` 读取
- 自动收敛到最优解
- 输出优化历史到 `optimization_history.json`

## 两个入口的区别

| 特性 | app.py | bayesian_optimizer.py |
|------|--------|----------------------|
| 参数来源 | 用户编写的 JSON | 贝叶斯算法自动生成 |
| 优化方式 | 顺序执行 | 智能搜索 |
| 适用场景 | 已知参数序列、调试 | 未知参数空间、自动寻优 |
| 灵活性 | 高（精确控制） | 低（自动） |
| 需要准备 | steps.json | 无需额外准备 |


## 配置文件

| 文件 | 用途 |
|------|------|
| `analysis_config.json` | 图像分析参数、文件路径 |
| `camera_cap_config.json` | 相机采集配置 |
| `gun_modbus_config.json` | 串口配置、参数范围 |

## 常见问题

- **图像采集失败**: 检查相机连接、SDK路径
- **Modbus连接失败**: 检查串口COM号、波特率
- **score异常**: 查看 `out/single_debug.png` 调试图