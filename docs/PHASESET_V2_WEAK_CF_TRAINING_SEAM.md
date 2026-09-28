# PhaseSet-V2：独立弱监督训练接口资格

2026-09-27。按 [用户关系监督修订](PHASESET_V2_RELATION_SUPERVISION_AMENDMENT.md)
实现软件接口，不再等待两位新标注者。不改变官方人工主检索 gallery、最终
test、87 阶段矩阵、三 seed 或 300 GPU-hours 上限。

## 已实现的边界

`ParentWeakCounterfactualRows` 使用独立 `included_weak` mask，没有
`verified_false` 字段。正例必须是同一 parent 的原始人工行，负例必须追加在
训练 text pool 的人工 gallery 前缀之后；与该 parent 原人工正例完全相同的
句子不能作为弱负例。索引和 mask 检查不证明一个句子在真实视频中为假。

`weak_counterfactual_loss` 保留已登记 softplus margin 公式，独立于原人工
verified-false 入口。`weak_coordination_objective` 的 variable-positive
symmetric InfoNCE 只使用原人工前缀；额外弱句子不成为检索正例或主 gallery。
基础模型仍拒绝 CF 训练输入。

完整 parent cache/VJP/backward 和原 replay/RNG 路径支持独立弱行类型。
`ParentBackwardResult.weak_counterfactual_count` 和主机 `weak_cf` 事件与原
`verified_cf` 分开。它统计纳入的弱行数，不是独立真值数量；A8 的 CF weight
为 0 时仍记录相同输入池的纳入数量，但不贡献 CF loss。旧六字段位置构造
保持兼容，TMR/WaMo/MIME 的新 weak count 默认 0。

## 一次真实服务器软件检查

Fresh ARIS GPT-5.6-Sol/xhigh 受限复核无 blocking/non-blocking，仅同家族
provisional。随后只运行一次隔离尝试，2026-09-27 17:48:16–17:51:30 UTC：
CPU FP32、单计算线程/单 interop、nice 10、离线、无 CUDA、16 GiB 上限和
900 秒 timeout。warm environment 没有重建，不分配或抢占其他 GPU。

**37 passed、0 skipped、0 failures/errors，pytest 189.61 秒。** 31 项新测试
加 6 项显式原接口检查，覆盖：

- margin/mask 公式和梯度、全排除 exact zero、错误 source/column/type 拒绝；
- 原人工前缀 InfoNCE、弱句子隔离、完整 dense-vs-replay 参数梯度一致；
- 真正 analytic AdamW 主机更新、中断/resume 的 state/optimizer/scheduler/RNG
  一致、弱与人工计数分离，以及 validation gallery 污染产生 FAILED terminal；
- TMR/WaMo/MIME 既有 analytic 主机端点对新增事件字段的兼容性。

包含一次 Torch nested-tensor prototype 警告，无异常；三 exit code 均为 0，
stderr 空，source/invocation 前后逐字一致。19 个 hashed metadata 成员及
receipt 共 20 个 flat 文件双端核验；仅元数据私有留存，没有公开数据或权重。
Receipt SHA-256：
`37ec17cb43d3c7706942cf7861c4a3753a415205dd9b9ab2789e30346e7f4ac7`。

生产预算三个原事件及 summary 前后不变。正式 **0/87**、pilot **0/12**，历史
GPU 使用仍 34 秒，本次新增 0 秒。测试中的 analytic optimizer 是实际执行，
不是 native 训练、合格 pilot、收敛或论文检索成绩。

## 未完成与不能声称的内容

该接口不生成弱句子、不验证它们的来源/语义，也不把现有人工正例当成 false
CF 的证明。实际训练池、固定生成 prompt/选择规则、私有生成记录和 CLIP
编码准入仍待完成；当前 `ParentHumanClipRows` 继续只返回原人工行和空 human
CF，没有偷偷注入机器标签。

这些检查不关闭真实 native CF/yaw、GPU training profile、预算内 pilot 和
87 阶段冻结/训练 gate，不证明任何关系理解或原人类挑战成功。

公开重现本页的 37 项选择（在合适的独立 CPU 环境中）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:tests python -m pytest -q tests/test_parent_weak_cf.py \
  tests/test_continuous_parent_training.py::test_complete_gallery_replay_matches_dense_loss_and_all_parameter_gradients \
  tests/test_continuous_parent_training.py::test_rng_advances_once_and_mode_restores_after_complete_capture_replays \
  tests/test_continuous_parent_host.py::test_host_runs_optimizer_selects_validation_and_preserves_full_parent_census \
  tests/test_continuous_base_retrieval.py::test_base_host_stage_checkpoint_and_cf_contracts_are_explicit
```

上述命令不包含私有预算 caller，不会重新建立真实训练准入或数据来源证据。
