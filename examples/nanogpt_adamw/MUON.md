# w768 Muon + AdamW 正式训练

独立入口 `train_w768_muon.sh` / `launch_muon.py`，原 AdamW 入口保持不变。复用同一模型、FineWeb IndexedDataset、官方 `pretrain_gpt.py` 和 checkpoint 流程，不包含本地小模型测试分支。

## 优化器配置

| 参数类别 | 优化器 | 峰值学习率 | weight decay |
|---|---|---|---|
| Transformer 二维权重 | Muon | 0.00036 | 0.1 |
| 所有一维 Norm gamma | AdamW | 0.0018 | 0 |
| 共享 embedding/output 权重 | AdamW | 0.0036 | 0.1 |

Muon momentum=0.95，Nesterov=True，Newton–Schulz 5 次；固定系数 `(3.4445, -4.7750, 2.0315)`，在官方接口对应 `coefficient_type=simple`。Q、K、V 分开正交化。更新量乘 `sqrt(max(rows, columns))`，对应官方 `scale_mode=spectral`、`extra_scale_factor=1.0`，没有额外 0.2 系数。AdamW beta=(0.9, 0.95)、eps=1e-8。

配方来源为旧脚本 `train_small_batch_muonw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0p1_wsd_seed00.py`，SHA256 `a01c2fdd4032102bc7b320cafc7961d83604117b7c12746ce931293d372bd9eb`。这里只迁移优化器配方，训练规模沿用当前正式配置：12层、width768、6头、seq1024、8卡、每卡64、global512、5100步；WSD预热250、平台3400、线性衰减1450至0。

## 云端运行

使用原 AdamW 已运行的 Megatron 环境与相同上游版本，额外确保安装上游固定的 Emerging Optimizers v0.2.0：

```bash
python -m pip install 'emerging_optimizers @ git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0'
bash examples/nanogpt_adamw/train_w768_muon.sh \
  --data-dir /path/to/fineweb/megatron \
  --output /path/to/runs/w768_muon
```

`--dry-run` 只打印命令；恢复同一 Muon 实验时使用原 output 加 `--resume`。必须为 Muon 使用新的输出目录，不能接着 AdamW checkpoint 运行。数据准备方法见 [README](README.md)。

`launch_muon.py` 直接启动官方 `pretrain_gpt.py`。在 `megatron/training/training.py` 的 `get_megatron_optimizer_config()` 中，Muon 条件分支增加一维参数的学习率/WD规则；没有运行时函数替换。该本地定制影响本 checkout 的所有 Muon 实验，一维参数在当前无 bias 模型中是 Norm gamma。云端部署时除了 examples 文件，还必须同步这处 training.py 修改（相同固定上游版本）；仅复制 examples 不足以复现配置。启动器会备份修改后的 training.py。

## 验证范围

2026-10-16：Windows 单 GPU、local 实现、FP32、2层/width128/seq32/global4，真实 FineWeb 数据运行5步、验证、保存成功。连续5步和2步后恢复至5步，模型、优化器和调度器状态逐项完全一致。checkpoint 中三类参数的实际优化器、峰值学习率、WD与上表一致。

本次没有执行云端8卡、w768、Transformer Engine/BF16训练。官方实现使用 FP32 归一化、EMA 形式动量和 BF16 NS 运算；与旧脚本的计算顺序及 epsilon 处理有差异，因此这里对齐配方，不宣称旧脚本逐位数值等价。

## 可选固定矩阵 RMS（2026-11-03）

在原启动命令追加 `--fixed-matrix-rms`，使用新 output。此开关将 WD 设为0；学习率、WSD、Muon NS配方不变，不做ELR控制。不加开关仍为原WD=0.1版本。

主干矩阵与共享embedding在每次底层优化器成功step后投影到各自初始RMS，Q/K/V按逻辑矩阵分别处理，一维gamma不参与。通过正式step post hook在FP32主权重上操作，然后沿用Megatron的主权重到BF16模型同步。目标保存在optimizer param_groups中，恢复时验证布局并加载原目标，不能用普通Muon checkpoint直接恢复。

支持范围：当前dense Muon、普通DP、均匀连续TP分片、PP=CP=1、非分布式优化器、torch checkpoint；不支持PP共享权重、MoE、FSDP/分布式优化器、GTP重分片或strided TP。TP组来自框架；仅归约矩阵平方和和元素数，不收集完整矩阵。固定范数不会改变Muon的tp_mode；以后使用TP>1时仍需明确选择Muon算法模式。

本地验证：4个CPU/Gloo进程模拟TP2×两组独立副本，行/列分片与融合QKV投影对照完整矩阵通过，恢复一致；这是投影通信验证，不是完整多卡模型训练。真实FineWeb单GPU local实现FP32/BF16小模型5步通过；FP32 2+3恢复与连续5步模型/优化器/调度器完全一致，BF16主权重13个逻辑矩阵最大相对RMS误差约7.42e-8。未验证多机NCCL、TE、完整TP训练或吞吐。

实现见 `megatron/core/optimizer/fixed_rms.py`，安装入口在training.py。部署需先应用 muon-gamma.patch，再应用 muon-fixed-rms.patch，具体命令见仓库根README。已应用gamma补丁的环境只需追加fixed-rms补丁；不能只复制启动脚本。

### 完整链路模拟补充验证
2026-11-03：同一CUDA GPU上的双进程运行完整TP2小模型（DP1、PP1、BF16、local实现、Muon distributed模式），集合通信由测试适配器经CPU/Gloo同步中转。真实数据5步与2+3恢复通过；每步13个逻辑矩阵全局RMS独立重建检查通过，最大相对误差9.642e-8。每步FP32主权重到BF16模型复制精确核对通过；各rank模型/优化器/调度器恢复结果逐位一致。TP1 BF16同样通过。这是功能模拟，不是双物理GPU、NCCL、TE或性能验证；正式代码未加入测试适配器。
