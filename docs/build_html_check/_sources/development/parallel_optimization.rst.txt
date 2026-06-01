并发调度优化策略
================

本文档记录 HABIT 项目 ``habit/utils/`` 并发子系统的调度优化策略、实现细节与性能影响。

背景
----

HABIT 的生境分析流水线需要处理大规模队列（数百至数千受试者），每个受试者的
Stage-1（个体级特征提取）是计算密集且内存敏感的操作。项目已经构建了两种并行执行模式：

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - 模式
     - 核心类
     - 特点
   * - **isolated**
     - :class:`~habit.utils.isolated_runner.IsolatedTaskRunner`
     - 每个 item 启动独立 spawn 子进程；隔离性最好，但 spawn 开销大
   * - **persistent**
     - :class:`~habit.utils.persistent_worker_runner.PersistentWorkerPoolSession`
     - 长驻 worker 池，通过 task_queue 派发任务；减少 spawn 开销

两种模式共用 GPU slot 管理（:mod:`~habit.utils.parallel_gpu_utils`）、
checkpoint/resume 和 auto-retry 机制。

本次优化针对调度层延迟和资源弹性两个维度，在不改变外部接口的前提下提升大队列场景的吞吐。

优化 1：Event 驱动唤醒（isolated 模式）
---------------------------------------

问题
^^^^^

``IsolatedTaskRunner.map_items`` 的主轮询循环使用 ``time.sleep(0.25)`` 固定间隔。
每个 item 完成后最多等 250ms 才被主循环感知，500 个 subject 累计浪费约 125s。
``_fill_slots()`` 也只在轮询周期后触发，新任务派发同样有 250ms 延迟。

解决方案
^^^^^^^^

在 :class:`_ActiveSlot` 中引入 ``threading.Event`` 字段 ``result_ready_event``：

.. code-block:: python

   @dataclass
   class _ActiveSlot:
       result_ready_event: threading.Event = field(default_factory=threading.Event, repr=False)
       # ... 其余字段不变

Reader thread 收到结果后立即设置 event：

.. code-block:: python

   # _drain_worker_result_queue 内部
   recv_bucket[0] = result_queue.get(timeout=poll_interval_sec)
   if recv_bucket[0] is not None:
       slot.result_ready_event.set()

主循环用 ``_wait_for_any_result`` 替代固定 sleep：

.. code-block:: python

   def _wait_for_any_result(active_slots, max_wait_sec):
       deadline = time.monotonic() + max_wait_sec
       while True:
           remaining = deadline - time.monotonic()
           if remaining <= 0:
               return
           for slot in active_slots:
               if slot.result_ready_event.is_set():
                   slot.result_ready_event.clear()
                   return
           time.sleep(min(remaining, 0.05))

效果
^^^^^

- 每个 item 完成后的感知延迟从 ~250ms 降至 ~50ms
- 500 个 subject 的大队列可节省约 80s 等待时间
- 完全向后兼容，不改变外部 API

相关文件
^^^^^^^^

- :file:`habit/utils/isolated_runner.py` — ``_ActiveSlot``、``_drain_worker_result_queue``、``_wait_for_any_result``

优化 2：Persistent Worker Pool OOM Backoff
-------------------------------------------

问题
^^^^^

Isolated 模式有 ``oom_backoff`` 机制：遇到 ``MemoryError`` 时自动减少
``max_workers``，让后续 subject 有更多内存空间。但 persistent 模式只在
``_finish_result`` 中重启 worker slot，**不减少活跃 worker 数量**，在内存
压力下仍可能级联 OOM。

解决方案
^^^^^^^^

为 :class:`PersistentWorkerPoolSession` 添加与 isolated 模式对齐的 OOM backoff：

.. code-block:: python

   class PersistentWorkerPoolSession:
       def __init__(self, ..., oom_backoff=True, oom_reduce_workers_by=1):
           self.oom_backoff = oom_backoff
           self.oom_reduce_workers_by = oom_reduce_workers_by
           self._effective_max_workers: int = max_workers

       def _apply_oom_backoff(self, item_id, *, logger=None):
           if not self.oom_backoff:
               return
           new_max = max(1, self._effective_max_workers - self.oom_reduce_workers_by)
           if new_max < self._effective_max_workers and logger is not None:
               logger.warning(
                   "Fatal memory error for item %s in persistent pool; reducing "
                   "effective parallel workers %s -> %s",
                   item_id, self._effective_max_workers, new_max,
               )
           self._effective_max_workers = new_max

调度逻辑使用 ``_effective_max_workers`` 限制并发 busy slot 数：

.. code-block:: python

   while pending_items and idle_slots and n_busy < self._effective_max_workers:
       slot = idle_slots.pop(0)
       _dispatch_to_slot(slot)
       n_busy += 1

参数传递链
^^^^^^^^^^

YAML 配置 → ``HabitatAnalysisConfig`` → ``IndividualCheckpointStage._parallel_kwargs()``
→ ``PersistentWorkerPoolSession(oom_backoff=..., oom_reduce_workers_by=...)``

效果
^^^^^

- Persistent 模式与 isolated 模式行为一致：OOM 后自动降低并发度
- 避免多 worker 同时 OOM 导致整个批次失败
- ``_effective_max_workers`` 只影响当前 ``map_items`` 调用，不修改原始 ``max_workers``

相关文件
^^^^^^^^

- :file:`habit/utils/persistent_worker_runner.py` — ``_apply_oom_backoff``、``_effective_max_workers``
- :file:`habit/utils/parallel_utils.py` — 传递 ``oom_backoff`` / ``oom_reduce_workers_by``
- :file:`habit/core/habitat_analysis/checkpoint/stage.py` — ``_ensure_persistent_pool`` 传递参数

优化 3：自适应 queue.get 超时
------------------------------

问题
^^^^^

``PersistentWorkerPoolSession.map_items`` 的主循环使用
``result_queue.get(timeout=0.25)`` 固定超时。当 per-item timeout 接近到期时
（例如还剩 0.05s），主循环仍需等完整 250ms 才能检测到超时。

解决方案
^^^^^^^^

新增 ``_next_deadline_wait_sec()`` 方法，计算最近一个即将到期的 deadline，
返回距离该 deadline 的时间（上限 0.25s）：

.. code-block:: python

   def _next_deadline_wait_sec(self) -> float:
       max_wait = DEFAULT_POLL_INTERVAL_SEC  # 0.25
       if self.per_item_timeout_sec is None:
           return max_wait

       now = time.monotonic()
       for slot in self._slots:
           if not slot.busy or slot.started_at is None:
               continue
           remaining = (slot.started_at + self.per_item_timeout_sec) - now
           if remaining <= 0:
               return 0.001
           if remaining < max_wait:
               max_wait = remaining

       # 也检查 awaiting_ready 的 startup timeout
       if self.spawn_startup_timeout_sec is not None:
           for slot in self._slots:
               if slot.ready or slot.awaiting_ready_since is None:
                   continue
               remaining = (slot.awaiting_ready_since + self.spawn_startup_timeout_sec) - now
               if remaining <= 0:
                   return 0.001
               if remaining < max_wait:
                   max_wait = remaining

       return max(0.001, max_wait)

主循环使用自适应超时：

.. code-block:: python

   wait_sec = self._next_deadline_wait_sec()
   reply = self._result_queue.get(timeout=wait_sec)

同样优化了 ``start()`` 方法中的 startup 阶段等待，基于自身 startup deadline。

效果
^^^^^

- Timeout 检测精度从 ±250ms 提升到 ±1ms
- 避免在 deadline 已过后的无效等待
- 短 timeout 场景（如测试用例或紧急中断）受益最大

相关文件
^^^^^^^^

- :file:`habit/utils/persistent_worker_runner.py` — ``_next_deadline_wait_sec``、``map_items``、``start``

优化 4：降低 _poll_dead_workers 调用频率
-----------------------------------------

问题
^^^^^

``map_items`` 主循环每次迭代都调用 ``_poll_dead_workers``，遍历所有 slot
的 ``proc.is_alive()``。在快速结果到达的场景下（如特征提取很快的小 subject），
主循环迭代频率很高，每次都做 ``is_alive()`` 检查是不必要的系统调用开销。

解决方案
^^^^^^^^

添加 ``_last_dead_worker_poll_at`` 时间戳，限制主动调用频率为每 0.25s 一次：

.. code-block:: python

   # 主动循环中——限频调用
   now_for_poll = time.monotonic()
   if now_for_poll - _last_dead_worker_poll_at >= DEFAULT_POLL_INTERVAL_SEC:
       self._poll_dead_workers(...)
       _last_dead_worker_poll_at = now_for_poll

在 ``queue.get`` 超时（Empty）时强制触发 poll——因为此时无结果到达，
需要检测意外死亡的 worker：

.. code-block:: python

   except Empty:
       self._poll_dead_workers(...)
       _last_dead_worker_poll_at = time.monotonic()
       continue

效果
^^^^^

- 主循环因快速结果到达而频繁迭代时，避免每次遍历所有 slot 做 ``is_alive()``
- ``Empty`` 分支仍保证死亡 worker 能被及时检测
- 对大 worker 数（processes=8+）的优化效果更显著

相关文件
^^^^^^^^

- :file:`habit/utils/persistent_worker_runner.py` — ``map_items`` 主循环

未采纳的优化方向
----------------

以下优化方向经评估后暂不实施：

per-slot pipe + select 替代共享 result_queue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

将每个 slot 的 result 改为独立的 ``multiprocessing.Pipe``，主循环用 ``select``
同时监听所有 fd。可消除共享 Queue 的锁竞争，但：

- Windows 上 ``select`` 仅支持 socket，不支持 pipe fd
- 改动涉及 IPC 协议变更，影响面大
- 当前共享 Queue 在 2-8 worker 规模下锁竞争不构成显著瓶颈

IPC 共享内存 / 写磁盘替代 pickle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``multiprocessing.shared_memory`` 或 worker 直写磁盘（parquet）来传递大型
特征矩阵，避免 pickle 序列化开销。但：

- 现有的 ``CheckpointSaveStep`` 已经在子进程写磁盘（pkl），属于部分实现
- 完全替代需要重构 ``ProcessingResult`` 的数据流，影响所有上游调用者
- 收益主要体现在极大特征矩阵（>100万体素）场景，一般队列不显著

I/O 与 GPU 计算重叠（双缓冲预取）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 persistent worker 内部实现双缓冲：当前 subject 在 GPU 上做特征提取的同时，
预取下一个 subject 的图像数据。但：

- 需要改造 ``persistent_worker_main`` 的 run loop，允许在等待 GPU 结果时
  提前拉取下一个 item
- 当前 worker 严格串行处理（取 item → 执行 → 放回结果），改为双缓冲需要
  管理 item 预取状态和错误回滚，复杂度高
- 收益取决于 I/O 占比，GPU-bound 场景收益有限

lazy item 消费
^^^^^^^^^^^^^^

将 ``parallel_map`` 的 ``list(items)`` 改为按需消费，减少大队列的内存峰值。
但与现有的 checkpoint/auto-retry 机制冲突——``_resolve_pending`` 和
``_run_parallel_pass`` 需要知道完整 pending 列表。改动面太大，收益有限。

配置参考
--------

与并发优化相关的 YAML 配置项：

.. code-block:: yaml

   processes: 2
   individual_subject_timeout_sec: null
   individual_subject_parallel_mode: persistent
   oom_backoff: true
   individual_subject_auto_retry_rounds: 2
   persistent_worker_max_consecutive_failures: 1
   persistent_worker_recycle_after_tasks: 0

配置说明：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 配置项
     - 说明
   * - ``processes``
     - 并行 worker 数量（自动被 GPU 池大小限制）
   * - ``individual_subject_timeout_sec``
     - 单个 subject 的墙钟超时（null 禁用）
   * - ``individual_subject_parallel_mode``
     - ``isolated`` 或 ``persistent``
   * - ``oom_backoff``
     - OOM 后是否自动降低并发度
   * - ``individual_subject_auto_retry_rounds``
     - 同次运行中自动重试失败 subject 的轮数
   * - ``persistent_worker_max_consecutive_failures``
     - 连续失败多少次后重启 worker slot
   * - ``persistent_worker_recycle_after_tasks``
     - 成功处理多少个 item 后回收 worker（0 禁用）

测试验证
--------

运行并发子系统的测试套件：

.. code-block:: bash

   python -m pytest tests/utils/test_isolated_runner.py \
                      tests/utils/test_persistent_worker_runner.py \
                      tests/utils/test_parallel_gpu_utils.py \
                      tests/habitat/test_checkpoint.py \
                      tests/habitat/test_checkpoint_stage.py \
                      -v --tb=short

版本历史
--------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - 版本
     - 日期
     - 变更
   * - v1
     - 2026-05-22
     - Event 驱动唤醒、OOM backoff、自适应超时、poll 限频
