# PhaseSet-V2：训练-only缓存弱文本池

2026-09-27。按 [关系监督修订](PHASESET_V2_RELATION_SUPERVISION_AMENDMENT.md)
继续使用原官方人工主检索文本，机器句子只能作为单独披露的弱训练监督。

## 缓存与阶段隔离

`pool_frozen_clip_text_rows` 精确选择两个 original CLIP batch，再追加机器
向量。人工前缀按位不变，不重新编码，不消耗 RNG。派生 receipt 保存两个
original receipts、各自原行索引及人工前缀长度，明确不是一次新的 CLIP 编码。
不同模型/snapshot/runtime、重复 commitment 和篡改 ancestry 被拒绝；原始
rehydrator 不接收派生池。两个 original 的 source provenance 均保留，不
虚构共同 encoder source。commitment 只用于谱系，不进入模型。

`ParentWeakClipRows` 绑定同一完整开发任务的人工 provider，显式接收当前
阶段 training components、原机器编码 receipt 和每句来源记录。检查 source/
family/人工 ordinal、政策/生成记录摘要及编码文本相符；机器文本不能是同一
parent 的原人工正例。阶段 held-out 不进入训练，外层 fold 的训练组件可以
包含旧主验证组件，不混淆这些不同任务。**这些检查不证明机器语义正确。**

训练只选择当前 batch 对应的弱句子并追加于全部原人工 occurrence 后。
没有候选的 parent 返回空 weak rows，不补造标签。evaluation 直接走原
`ParentHumanClipRows`，不读机器池；基础模型在读取 motion 前拒绝 weak CF。
既有 disk source 接入为显式可选项，原 human-only 路径保留。

## 一次真实服务器软件资格

Fresh ARIS GPT-5.6-Sol/xhigh 部署前受限复核 PASS/no findings，同家族
provisional。2026-09-27 18:32:25–18:33:47 UTC，仅运行一次：单线程 CPU
FP32/interop 1、nice 10、离线/无 CUDA、16 GiB/900 秒限额，warm env 未改变。

**24 passed，0 skipped/failures/errors，76.80 秒。** 21 项新用例加 3 项
相关原人工接口用例，覆盖缓存前缀、honest receipts、阶段过滤、独立计数、
全人工 occurrence、重排、排除 mask、空候选、disk source、validation
不污染，以及 analytic AdamW 完整主机中断/resume bitwise 一致。

三 exit code 均 0、stderr 空、source/invocation 前后不变。20 个 flat
metadata 文件（19 hashed 加 receipt）双端核验，receipt SHA-256：
`6074d028be82ab5564f3577044190f9f0496769ae1d657f33e028d5c34b464a0`。
原生产三个预算事件与 summary 不变；正式 **0/87**、pilot **0/12**，本次
GPU 0 秒。analytic optimizer 用例不是原生学习或论文结果。

## 后续真实训练文本准备（不是训练结果）

同日完成一次固定 pilot-learning 文本批次：34 个训练 parent、169 条原官方
人工 occurrence，GPT-5.6-Sol 纳入 57 条弱句子、排除 112 条。仅读取这些
parent 的人工 holistic 文本；不读取 motion、phase、模型分数、held-out
文本或 test。每 occurrence 只允许一次候选或排除，不按配额补齐，不做
语义重生成。一次无语义输出的真实网络超时保留证据，修复传输后单次恢复；
首个成功 job 在完整批次复用。原始私有生成记录和完整批次均已冻结。

真实 CPU CLIP 准备于 19:40:37–19:42:15 UTC 完成：只编码这 57 条新增
机器句子，未截断、最大未截断长度 71 tokens。原 1,334 条人工向量没有
重新编码；当前 169 条训练人工前缀按位不变。全部 253 个开发 parent 的
evaluation 仍只返回原人工文本，stage held-out 不进入 weak training。
模型、RNG、源码、资产及生产预算前后不变，所有退出码 0、stderr 空。

这是数据准备，不是 native learning、pilot 或检索增益。训练 yaw、实际
native CF backward、GPU profile 及正式实验仍待；正式 **0/87**、pilot
**0/12**。机器标签永不升级为人工
`verified_false`、独立运动真值或历史两人挑战成功。

公开选择重现（独立合适 CPU 环境，不含私有预算 caller）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:tests python -m pytest -q tests/test_parent_weak_clip_rows.py \
  tests/test_parent_clip_rows.py::test_selection_is_exact_owned_and_honest_without_encoder_runtime_or_rng \
  tests/test_parent_clip_rows.py::test_every_human_occurrence_and_reordered_parent_batch_are_preserved \
  tests/test_parent_clip_rows.py::test_disk_source_binds_the_same_task_and_returns_explicitly_no_cf
```
