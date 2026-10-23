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
