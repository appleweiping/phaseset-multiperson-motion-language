# PhaseSet-V2 研究修订 — 2026-09-26

2026-09-27 后续用户指令已形成
[关系监督修订](PHASESET_V2_RELATION_SUPERVISION_AMENDMENT.md)：不再等待两位新
标注者，已有人工主检索标签保持不变；机器关系监督只作披露的弱监督/探索性
证据，不能替代下文历史的人类挑战成功条件。其接入仍须实际实现和资格。

此修订执行用户指定的“升级至顶会程度”聊天。研究目标是区分
“个体动作相近、协同关系不同”的原生多人场景。旧协议和历史证据保留；
受影响的正式训练使用本修订的87阶段矩阵，不再叠加旧33-run矩阵。

## 固定的三个升级

1. 方向保持的局部相位：保留带符号的XYZ关节速度，区分pelvis运动与
   身体相对pelvis运动。沿用既有20Hz六频带核，提取局部复交叉响应、
   coherence、端点能量和支持。低能量相位标记为不可观测。
2. 时间关系：保留actor–edge incidence，沿同一个人物及同一条边的时间轨迹
   编码，再做集合读出；完整capture按实际窗口起始时间做有序读出。
3. 协同语言监督：匹配完整、端点绑定的关系表示，使用variable-positive
   InfoNCE及有核实假标签的反事实softplus损失。人物重编号不是负例；
   泛泛动作描述在相位扰动后仍可能为真，不能自动标假。

评分单独校准：两分支先变为cosine，在共同学习尺度上以sigmoid权重混合。
A9使用旧表示、只改变校准，不能把尺度修复归为新机制贡献。
A6对平均与差分信号先做DCT，保留交叉项；不以端点自功率平均冒充它。

坐标约定为共享世界坐标。世界方向反相不自动等于面对面镜像手势的
语义反相。所有相位操作仍由motion产生，文本不能更改物理字段。
全人对时间复杂度为O(K²)；流式显存不代表线性时间。

## 实验与停止边界

| 阶段 | 正式训练阶段 |
|---|---:|
| B0/B1/B2，三seed | 9 |
| TMR-Set/WaMo-Set/MIME-Set，三seed | 9 |
| 旧PhaseSet与V2 residual，三seed | 6 |
| A1–A9，各三seed | 27 |
| 三group-held-out folds，各三seed的B2、V2、冻结强对手 | 27 |
| Inter-X：三seed的B2、V2、原MIME | 9 |
| 合计 | 87 |

机器矩阵见 [配置](../configs/phaseset_v2_experiment_matrix.json)。
seeds为1729/2718/31415。base最多30epoch、residual最多20epoch；
最多12短pilot、两轮结构返工、全局三次有根因的故障重启，
正式启动最多90，总保护预算300 GPU-hours。先profile代表性任务，再冻结
max_steps；每个pilot不超过对应正式阶段20%的optimizer steps。
默认pilot为三个文献家族各3次、B2一次、V2 head两次，总计12次。
只允许冻结前一次有依据的统一调整；epoch可统一扩大至base60/residual40，
但仍受300 GPU-hours和已冻结max_steps约束。达到限制后交付真实终态。

性能比较保留文献原架构及辅助损失的合理多人适配；机制比较固定B2、
文本来源及相近有效容量（head参数默认差异不超过5%，不计无用dummy参数）。
TMR普通双塔不能替代完整TMR，WaMo功率头不能
替代原方法，MIME多人适配及收敛情况须明确。外折独立训练，权重不跨折复用。
Inter-X如在其训练集训练，只能称外部双人适用性验证。

A1速度输入、A2无时间顺序、A3pair-bag、A4错误incidence、A5无显式phase、
A6真实mean/difference DCT、A7相同反事实数据的普通局部关系匹配、
A8无反事实目标、A9只校准旧表示。每一贡献必须有对应可推翻它的对照。

## 数据和科学验收

保留400/96/76原split，主validation为C00，最终test封存C09/C11/C15。
用户提供的`PhaseSet_Codex_Handoff_2026-09-26.md`已接入：pilot训练C01/C02、
验证C03；fold0留出C00/C06/C10/C14，fold1留出C04/C07/C12，
fold2留出C05/C08/C13。各折训练开发组件中除C03及本折held-out的所有组件，
checkpoint selection统一C03，不复用主实验checkpoint。
冻结强对手B*只能按pilot验证从TMR-Set/WaMo-Set/MIME-Set选择。
没有观察到真实数据时，不执行正式训练或生成主表数字。

真实关系挑战至少200核实最小文本对、100capture、8参与者组件，
由两名不知道模型分数的人工标注者审核。不能以机器改写凑数。
先确认原生数据中的可观察关系是否足够；不自动追加Multi-TPC。

内部科学目标：主双向R@1三seed均值比最强合格文献baseline至少+2pp；
关系挑战至少+5pp；三个外折平均效应为正；方向/结构/反事实对应消融
支持并进行预登记多重比较；超过A9；非周期相对B2退化不超过1pp；
相同硬件和缓存规则下端到端推理默认不超过强对手2倍。报告不确定性、
组件级结果及效率trade-off；这些目标不保证会议录用。

终态采用READY_FOR_SUBMISSION、COMPLETE_NOT_SUPPORTED、
BLOCKED_DATA/RIGHTS/RUNTIME/RELATIONAL_EVIDENCE/BUDGET或FAILED_IMPLEMENTATION。
软件测试、PDF编译和同模型家族审查不替代真实科学验收。
结论与正负结果一致；不追加seed、epoch、数据集或更换test追逐结果。

## 本次实现的边界

两个新模块实现固定物理前端、checkpointed时间incidence、
有序capture读出、共同尺度评分和核实负例损失。它们作为独立研究API，
尚未接入现有host正式训练、文献适配或数据标注。
handoff确认40帧统计/20帧hop，低频核支持约4秒，不能声称1秒精确定位。
起始时间编码为两层512D；两层状态可跨chunk连续，但host层的整capture
actor/edge轨迹、halo卷积和关系packet读出尚待接入，不能以窗口embedding
的有序读出冒充它们。评分起始alpha=0.1，无周期支持时显式回退global。
本次精确组件配置不等于pilot超参已冻结或正式矩阵可启动。
新前端的训练能量阈值需重新拟合；
旧五速度通道阈值不能直接套用。机制检查使用分析运动，不产生检索成绩。

正文与图表最终来自同一冻结aggregate，提供可编辑源、真实限制和失败案例。
本日查证 [ICASSP2027官方征稿](https://2027.ieeeicassp.org/call-for-papers/)
显示full-paper截止2026-09-23，已过。研究实现继续，投稿目标需按真实开放的
会议及当届模板/页数/披露规则确认。
