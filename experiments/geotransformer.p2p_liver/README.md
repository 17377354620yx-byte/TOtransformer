# P2P RTOR+A3 experiment

当前机器上的 Conda 环境名为 `geo_v3`（下划线）。旧的
[TRAIN_TEST.md](../../TRAIN_TEST.md) 仍记录 legacy/cooperative 实验命令。

`cooperative` 是未证实提升的实验候选，profile 保留原单骨干、RTOR、A3 和 LGR，取消 source 硬筛选，逐步引入 predicted coarse training proposals。`legacy` profile 用作原模型控制组。

新增的 attention-conditioning profile：

- `togg_phase1`：保留 GeoTransformer 原 distance/angular RPE，仅加入零初始化的局部 topology additive self-attention bias；不启用 overlap loss。
- `togg_phase2`：在 Phase 1 基础上，在每个 cross block 前预测 bilateral overlap，并加入逐层逐头的 `beta * log(p + eps)` key-side bias；`beta` 为零初始化。
- `togg_phase3`：增加 descriptor + learnable overlap prior + topology compatibility 的全局 coarse ranking；不启用 ranking loss，用作监督消融。
- `togg_full`：Phase 1–3 + budget-aware ranking loss，直接优化 GT correspondence 进入 Top-256。
- 所有 TOGGT profile 均关闭 hard overlap selection，不实例化 legacy 后置 RTOR。原 `legacy`、`cooperative` 和 `soft_overlap` 路径不变。

Phase 2 训练示例：

```bash
conda run --no-capture-output -n geo_v3 \
  python experiments/geotransformer.p2p_liver/trainval.py \
  --architecture rtor_a3 \
  --interaction_profile togg_phase2
```

推荐使用与 compact 实验相同的一体化脚本：

```bash
# 训练 150 epoch，然后依次测试 in-silico none/2/4 mm 和 in-vitro
bash scripts/run_p2p_togg_phase2.sh all

# 仅训练、断点续训、仅测试
bash scripts/run_p2p_togg_phase2.sh train
bash scripts/run_p2p_togg_phase2.sh train --resume
bash scripts/run_p2p_togg_phase2.sh test
```

测试其他 checkpoint：

```bash
P2P_SNAPSHOT=/absolute/path/to/checkpoint.pth.tar \
  bash scripts/run_p2p_togg_phase2.sh test
```

完整 Phase 1–4 统一从随机初始化训练并自动测试：

```bash
bash scripts/run_p2p_togg_full.sh all
```

同一完整模型 run 中断后可续训：

```bash
bash scripts/run_p2p_togg_full.sh train --resume
```

TOGGT 正式实验不加载 GeoTransformer、RTOR、Phase 2 或其他 checkpoint。
`--resume` 只用于续接相同 run 的 `snapshot.pth.tar`；一体化脚本会拒绝
`--warm_start` 和训练阶段的 `--snapshot`。

- `config.py`：架构与 interaction profile。
- `trainval.py`：训练、验证、best checkpoint。
- `test.py`：严格加载、RMS-TRE 和逐样本导出。
- `../../diagnostics/interaction/`：固定训练输入的结构、loss、梯度诊断。

完整诊断和适用边界见 [REPORT.md](../../diagnostics/interaction/REPORT.md)。

五阶段递进消融及运行命令见 [ABLATION.md](ABLATION.md)。

根据消融结果组织的精简模型及运行命令见 [COMPACT_MODEL.md](COMPACT_MODEL.md)。
