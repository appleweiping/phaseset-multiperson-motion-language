# V2 持久全局预算账本

`StudyBudget` 是单个研究的 append-only accounting journal，复用原 `_AttemptLease`
做短事务锁，不启动模型、SSH、GPU 或授权系统。不改 legacy33run 解析器。所有
生产任务必须使用同一个私有根目录；另建空账本或忽略历史违反研究合同。

初始化必须显式提供完整87行/三seed矩阵、每一个历史计数器、历史证据及每个
pilot家族对应的 nominal formal optimizer-step cap，不会默认历史为0。已有目录
不能重新初始化。GPU预算300h、formal reservations90、pilot12、rooted formal
restart3、结构返工2，以及五个pilot家族3/3/3/1/2的分配固定；不能通过传参放大。
Reservations是实际启动数的保守上界，spawn失败也消费名额，不伪装成成功训练。
当前研究尚无formal/pilot启动，因此bootstrap只允许显式确认的zero-training历史，
另导入真实既往GPU检查用量；若找到已有训练，必须沿用详细原journal，不能把
非零训练仅总结成几个计数后丢掉run/seed/predecessor身份重新初始化。

启动子进程前预约所有分配GPU乘以完整wall envelope（包括timeout和kill grace）。
未结算任务一直占用整份预约，进程缺失/过时heartbeat不自动返还。生产wrapper
仍须实际设置相同有界timeout，并确认自己的子进程已终止/不再持GPU之后才结算。
按实际分配wall time×GPU数收费，失败/中断也收费；早结束归还未用时长，但不
归还pilot/formal名额。超过预估/300h的实际用量照实结算，不能为使数字好看拒记。
Optimizer-step超cap也保留成本并标出，不能冒充合同通过；操作员须修复根因。
实际硬中断未留下完整terminal时，FAILED/INTERRUPTED可显式记录未知cursor=null，
成本仍照实计入；单列unknown cursor，不伪造0步，也不允许以未知cursor称COMPLETED
或NOT_STARTED。原整数cursor接口不变，重启仍需实际原始证据和checkpoint恢复。
账本不会自动kill进程、判断关系真值、选择seed或关闭科学acceptance gate。

profile必须0 optimizer updates；真实训练不能藏在未计数profile中。若完整training
profile执行真实优化器更新，就占既定pilot名额，同一次pilot可同时提供profile
证据，不再偷偷额外训练。每pilot `5*planned_steps <= nominal_formal_steps`，不以
epoch单位转换逃过20%限制。pilot编号唯一，失败名额不重置。正式run只匹配矩阵
ID和seed；先记录**已在外部真实冻结**的全87个max_steps，只可记录一次。
记录step limits本身不是研究freeze或训练许可；当前真实研究仍未完成该freeze。

Formal重复启动必须关联同run最新已结算失败/中断前驱，写真实原始日志根因，
全球只允许3次，不分配置额外给三次，也不从旧前驱分叉或重跑已完成seed。
GPU收费、starts/restarts、结构返工跨重新打开账本保存，不因新host或中断而清零。
达到三次formal rooted restart额度后，不再新增长formal任务；先交付实际故障证据
和预算缺口，不扩大额度。90 reservation上限也独立保护，未用seed不因此删掉。
写盘截断/gap等不完整journal会显式拒绝继续；保留证据，不能另建零账本重试。

JSON中的`scientific_or_launch_authority:false`明确是成本证据，不替代数据权利、
完整输入/环境/代码binding、true two-human CF、真实GPU空闲和dated科学合同。
生产外部controller的实际spawn/timeout/settlement仍要有真实收据，软件模拟账本
和测试中模拟90 reservations不等于启动了90个实验。所有数值/模型/资格仍只在
登记服务器；公开只含源代码、合成bookkeeping测试和不敏感说明。
新[Linux process controller](PHASESET_V2_STUDY_PROCESS.md)提供实际执行接入；
其本身不授科学admission，真实nativeGPU训练还需独立资格和数据/关系合同。

## 本次实际限定资格

Fresh ARIS GPT-5.6-Sol/xhigh发现首次建账的parent-directory durability缺口，
已修正为已有parent、event-file fsync → journal-directory fsync → parent fsync。
两个新增case验证POSIX顺序及sync失败时保留账本/拒绝重建；Windows不冒称同样
directory-fsync保证。最终限定复核none，same-family仍provisional。

登记服务器warmenv、CPU1/interop1/noCUDA/nice10/16GiB有界执行，新26项全部
PASS/0skip/0xfail，1.79秒；numerical/source-check/exit全0，源/invocation前后相同。
测试含并发最后额度保护、失败成本、pending保持、90合成reservation、三rooted
restart/两rework、12pilot分配/20%steps、截断journal与目录同步故障，不运行模型。

随后独立CPU-only历史导入真实核验四份既有CUDA收据：三个单卡mechanism检查
整段秒级receipt wall各加1秒rounding，保守合计34GPU秒；另一个profile在Torch/
CUDA分配前HOLD，计0。这是保守历史成本，不是精确allocation计时或新profile。
唯一私有生产journal现在history34、pending0、formal0、pilot0、formalstepfreeze
未记录。旧失败/资产/收据保留，未新增GPU分配/优化器步骤、重启或结构返工。

这些bookkeeping测试和一次历史bootstrap本身不替代production controller资格。
新controller另已完成[23项实际Linux软件资格](PHASESET_V2_STUDY_PROCESS.md)，
仍不替代native GPU接入、完整training profile、既定pilot/87阶段学习、two-human
关系核实、结果统计或论文交付。
