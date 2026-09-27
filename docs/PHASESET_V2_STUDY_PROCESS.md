# Linux owned-process执行与真实成本结算

`run_budgeted_process`把已实现的单一StudyBudget接到实际子进程生命周期：先读
GPU占用/其他compute，取得本研究per-GPU lease，再检查；完整wall envelope预约
后spawn前又查一次。不启动SSH、不使用shell=True、不自动重试，也不授科学许可。
UUID、visible cards、GPU数量和timeout+两份TERM grace+exit-verification envelope必须
相同；CPU1/PYTHONHASHSEED=0固定。CPU profile可无GPU；训练仍需真实外部admission。

每child新POSIX session，只有这个session被TERM/KILL。Linux WNOWAIT保留leader
PID直到所有同组live进程/CUDA持有者退出验证后再reap，避免PID过早回收；他人的
进程不终止。已登记训练程序不得daemonize/setsid逃出session；这是当前实际Python
parent host调用约定，不声称支持任意daemon或Windows生产服务。
子session由已存在的GNU timeout独立计时/TERM/KILL，父controller被SIGKILL也
不留下无限运行的child；账本仍保留其pending，不能因父死亡自动退款。controller
自身SIGINT/SIGTERM由main-thread暂存，拿到childhandle后先清理/结算，再恢复
原handler并报告中断；不在spawn中途因signal故意丢失handle。

正常退出后也检查遗留descendant；有遗留则清理并FAILED，不假称成功。忽略TERM
会在固定grace后KILL；不能证实退出、startup handle丢失或证据写盘失败时保留整
份pending预约，不猜PID/不自动退款。未来repair必须读原日志和实际状态，不盲重启。
只有确认termination之后，实测monotonic分配wall×全GPU数结算，失败/中断也收费。

`process-terminal.json`保存实际child exit、cleanup、host cursor和成本；单独
`settlement.json`只能在journal真的settle后出现。若结算后第二份收据写失败，
必须以实际journal为准，不重复收费。原primary错误不被cleanup/terminal secondary
覆盖。stderr/stdout和所有失败目录保留；没有动作/数据/模型资产公开。

`ContinuousParentTrainingHost`及继承它的`LiteratureParentTrainingHost`的
terminal `outcome/global_step`直接消费，不兼容被替代的legacy `Host` schema，不加载Torch
checkpoint猜optimizer进度。无/截断terminal记录cursor=null而非0、不得COMPLETED；
StudyBudget仍记录已知实际成本，并单列unknown cursor。它不把未知进度转换成
真实结果，也不为restart挑seed、设置floor/yaw或自动构造两真人CF许可。

软件验证只用真实stdlib子进程和合成journal；fake-card参数只测试账务，非实际
CUDA分配/硬件资格。实际GPU profile、完整native host/pilot、87datedfreeze、真人
关系核实及真实论文结果仍需要各自外部证据。当前没有新增正式/pilot训练。

## 登记服务器实际软件资格

2026-09-27一次CPU1/interop1/noCUDA/nice10/16GiB/warmenv有界执行，
23项全部PASS，10.90秒，0skip/0xfail；numeric/source-check/exit全0、stderr0。
19项新process cases及4项定向budget cases，真实stdlib child和信号，不是
CUDA训练或模拟日志替代真实进程。覆盖父controller被SIGKILL后的独立watchdog、
自身SIGTERM、忽略TERM、残留descendant、WNOWAIT、缺失/截断terminal、
三次GPU gate/lease碰撞、spawn歧义、PID/terminal/settle写盘及cleanup失败。
两个已有实际base/residual host terminal均固定SHA并成功读出真实global_step=4；
未重跑旧模型。整个资格前后同一生产journal摘要/原event SHA一致，无新增预约、
成本或native optimizer；所有故障仅发生于隔离合成fixture，原失败证据保留。
Fresh ARIS限定Sol/xhigh复核none，same-family仍provisional。Linux生命周期软件
资格已闭合；实际native训练仍0/87、pilot0/12，其接入/科学资格不因此闭合。
