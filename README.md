# Megatron w768 AdamW / Muon

本仓库存放在 NVIDIA Megatron-LM 上运行 w768 AdamW 实验的正式训练与数据转换脚本。训练循环、模型构建、优化器和 checkpoint 使用 Megatron 官方实现。

## 安装位置

这是脚本仓库，需配合 Megatron-LM 使用。已核对的上游版本记录在 `UPSTREAM.json`；不建议直接换成未经验证的最新版。

```bash
# 在已配置 Megatron 所需 CUDA/PyTorch/Transformer Engine 的 Linux 环境执行
cd /workspace
git clone https://github.com/NVIDIA/Megatron-LM.git
git -C Megatron-LM checkout cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54
git clone https://github.com/sant-liustu/megatron.git
cp -R megatron/examples/nanogpt_adamw Megatron-LM/examples/
cd Megatron-LM
```

依赖准备遵循该版本的官方 `skills/mcore-build-and-dependency/SKILL.md`。脚本不会自动安装依赖或下载数据。上游代码和依赖适用各自许可证，本仓库不包含 Megatron 核心代码。

## 数据转换

输入为 [kjj0/fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2) 的 GPT-2 token bin（FineWeb，而非 FineWeb-Edu）。原文件头1024字节，magic20240520、version1，载荷为uint16；转换不重新分词、不改变token。

```bash
python -m examples.nanogpt_adamw.prepare_data \
  --train /workspace/data/raw/fineweb_train_000001.bin \
  --valid /workspace/data/raw/fineweb_val_000000.bin \
  --output-dir /workspace/data/fineweb10B/megatron
```

输出目录须为空；生成 train.bin/train.idx、valid.bin/valid.idx 和 manifest.json。转换后官方读取器全量逐token核对。train和valid可分别列出多个分片。转换只需CPU，但依赖NumPy、PyTorch和Megatron。

## 训练

```bash
bash examples/nanogpt_adamw/train_w768.sh \
  --data-dir /workspace/data/fineweb10B/megatron \
  --output /workspace/runs/w768_adamw
```

- 加 `--dry-run` 只查看命令，不启动训练。
- 加 `--exit-interval 2` 在完整调度下运行2步并保存退出，可用于目标环境首次检查。
- 加 `--resume` 从同一output恢复；续训时去掉提前退出选项。
- 新实验使用新的output，避免混用不同配置的checkpoint。
- `--gpus`、`--micro-batch-size` 可显式调整，global batch必须整除总GPU数×micro batch。
- 多节点设置NNODES、NODE_RANK、MASTER_ADDR和MASTER_PORT，每节点执行同一命令，output使用共享目录；该路径尚未实测。

正式入口没有小模型模式、Windows测试适配或假数据回退，必须提供转换后的真实数据目录。

## 默认配置

| 参数 | 值 |
|---|---|
| 模型 | 12层、width768、6头、FFN3072 |
| 序列长度 / 词表 | 1024 / 50304 |
| GPU数 / 每卡batch / 全局batch | 8 / 64 / 512 |
| 梯度累积 / 更新次数 | 1 / 5100 |
| 实现 / 精度 | Transformer Engine / BF16 |
| 主干与Norm gamma峰值LR | 0.0018 |
| 共享embedding/output峰值LR | 0.0036 |
| AdamW betas / eps | (0.9,0.95) / 1e-8 |
| 矩阵weight decay / Norm gamma weight decay | 0.1 / 0 |
| WSD | 250步预热、3400步恒定、1450步线性衰减到0 |
| 梯度裁剪 / dropout | 关闭 / 0 |
| seed | 1234 |
| 日志 / 验证 / 保存间隔 | 10 / 500 / 1000步 |
| 每次验证 | 20个全局batch |

RMSNorm含Q/K可学习gamma，epsilon=1e-5；squared ReLU、默认RoPE、无bias、共享embedding/output。TP=PP=CP=1，其余为数据并行。总训练量2673868800 token。

已有token数据以NullTokenizer提供词表和EOD元信息（EOD50256），不经过文本重新编码。保持EOD参与loss，不按EOD重置attention/position。文本生成解码需要真正的GPT-2 tokenizer。

## 验证与边界

已在历史本地小模型上验证真实数据5步训练、验证与checkpoint保存；两个原始分片（train000001、val000000，各1亿token）转换后全部token一致，源文件SHA256匹配上游revision `889765ea1f903759787add96995d81171b632d0c`。另验证过官方训练的中断续训一致性。

当前正式8卡w768/TE/BF16、每卡batch64和多节点尚未实测；此前小模型结果不代表正式配置的显存、吞吐或稳定性。Megatron默认打乱样本，与旧nanoGPT顺序loader不保证逐batch一致。官方WSD初始更新LR0，与旧预热首步存在差异。仅一个1亿token训练分片跑满此配置会重复约26.7轮，正式比较需相同数据清单与token预算。

output保存argv、数据清单、Git状态/diff、源码副本、日志和官方torch checkpoint；数据、模型权重、运行日志和本地历史记录不包含在本仓库中。

## Muon 版本

保留原 AdamW 入口。Muon 配方见 [MUON.md](examples/nanogpt_adamw/MUON.md)。除复制 examples 外，必须将 gamma 分组规则补丁应用到上述固定版本的 Megatron：

```bash
# 在 /workspace/Megatron-LM 中执行；只应用一次
 git apply --check ../megatron/patches/muon-gamma.patch
 git apply ../megatron/patches/muon-gamma.patch
 python -m pip install 'emerging_optimizers @ git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0'
 bash examples/nanogpt_adamw/train_w768_muon.sh \
   --data-dir /workspace/data/fineweb10B/megatron \
   --output /workspace/runs/w768_muon
```

补丁仅修改 training.py：Muon 模式下的一维参数 LR=0.0018、WD=0；当前无 bias 模型中对应 Norm gamma。默认矩阵 Muon LR=0.00036、WD=0.1，embedding/output AdamW LR=0.0036、WD=0.1。此定制影响该 checkout 的所有 Muon 实验。启动器直接调用官方 pretrain_gpt.py，无包装入口。

本地单 GPU FP32 小模型真实数据5步、验证、保存及断点恢复已通过；尚未验证云端8卡w768 BF16 Muon。不能将该检查视为旧 nanoGPT 的逐位复现。
