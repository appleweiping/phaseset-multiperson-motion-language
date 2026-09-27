# 完整 parent host 的运行监测和 checkpoint 保留

`parent_run_monitor.py` 复用现有 `_AttemptLease`，不改旧 33-run 的
`AttemptStore`/run-ID 合同。`ContinuousParentTrainingHost.fit`（包含继承它的
TMR/WaMo/MIME host）在数值循环前持有 lifetime lease；终止/异常均释放。
每 30 秒写一次独立线程 heartbeat，包含当前操作、step、epoch、最近已发布
checkpoint SHA、单进程 monotonic elapsed。线程只读 scalar snapshot，不碰
模型、张量或 RNG，因此完整 capture forward/backward、数据读取和保存期间
没有新 update，也仍可判断进程存活。它不是每 30 秒必须完成一步的 watchdog。

现有不可变 run/events/checkpoint/resume 继续使用，新增不可变 heartbeat、
checkpoint metadata receipts 和 retention receipts。终止记录补充失败分类：
磁盘不足、checkpoint 无效、资源上限、进程中断、输入/合同错误或未分类。
原始 traceback 应留在私有日志；分类不授予自动重试。监测/终止写入同时失败时
尽量保留主异常并追加 note，不用清理错误替换原始训练根因。突然 kill/power loss
可能没有 terminal，最后 heartbeat 和 shell exit 收据需由操作员据实核对。

生产调用显式使用 `fit(..., checkpoint_keep_recent=2)`。默认 `None` 保留原先
keep-all 行为，不回收旧资格目录。保留最新两个 payload（回滚候选）、它们
各自依赖的 selected-best 以及当前 best；resume predecessor 在其他目录，不删除。
仅从本 attempt 自己发布并已记录的 checkpoint 中回收旧且不再被保留候选引用的
payload，逐文件确认目录/regular file/原摘要，先写 planned、删除后写 retired。
事件、摘要、历史 checkpoint receipts、旧失败目录均永久保留。不递归、不 glob
删除，不按未验证的磁盘占用猜测删除资产。已回收 payload 不能直接恢复，只能从
保留候选重新计算；planned 无 retired 明确是中途中断，不冒充已完成回收。

该监测器不批准正式实验、不实现全局预算或硬墙钟 timeout，也不把单进程 elapsed
冒充多 GPU-hours。私有执行器仍须对所有分配设备（包括失败计算）累计 GPU 时间，
在冻结的 87 阶段、最多 90 starts/12 pilots/3 rooted restarts/2 structural reworks/
300 GPU-hours 内预留和结算预算，并用有界 shell timeout 处理卡死操作。
监测器的存在不关闭真实 GPU profile、正式 freeze、逐 family yaw 或双人类 CF 证据。

资格测试只在登记服务器执行。分析性软件 optimizer/resume 通过不是 native
学习结果、短程 pilot 或论文效果；本实现也不修改 loss、优化器、调度器、
完整 parent/gallery 单位、验证选择或冻结 base 的算法。

登记服务器 CPU-only 资格于 2026-09-27 关闭：**19 PASS / 0 skip / 0 xfail**，
103.72 秒；包含新 16 case（base/residual 实际分析性 optimizer + bitwise resume、
初始化/heartbeat/terminal/release 故障、保留 best 依赖）及 3 个直接相关回归。
所有 numerical/source/invocation 检查 exit0，stderr 空；旧失败、旧资格不重启。
它不证明 GPU training profile、真实数据学习、正式预算或研究假设成立。
