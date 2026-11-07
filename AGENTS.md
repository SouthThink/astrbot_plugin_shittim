# 什亭之枢 AstrBot 插件开发指引

## 1. 定位

什亭之枢（Shittim）让 AstrBot 智能体拥有统一人格、跨会话记忆与主动行动能力：
所有私聊与群聊共享同一份记忆，她能自主决定**对谁说、在哪个会话说、什么时候说**。

一句话区分职责：**决断层决定「要不要说」，主脑决定「怎么说」，插件决定「说在哪里」。**

作息状态机、深睡期记忆整理（短期收口 / 名次降级 / 阶梯遗忘 / 清空账本）、历史回查（`search_history`）与推迟入睡（`delay_sleep`）均已落地。`plan/` 里目前没有待做项。

## 2. 目录与依赖

代码按**功能域**分包，目录名与「五条链路」一一对应：

```
core/
├── common/     通用基础设施：配置、数据库、事件取值、时间与分词、模型 JSON 解析
├── gate/       拦放链路：决断层 engine.py + 群聊巡检 patrol.py
├── focus/      焦点链路：manager.py
├── memory/     记忆链路：ledger / extractor / store / maintenance / profile
│               / timeline / injector / history / vision / forward
└── schedule/   作息链路：state.py

main.py         事件处理器、LLM 工具、生命周期
web_api.py      记忆 / 巡检 / 上下文页面接口
pages/memory/   记忆页面
```

- `core/common/` **不依赖任何业务包**，只被它们依赖。
- 新增模块先判断属于哪条链路；跨包引用一律用相对导入（`from ..common.db import MemoryDB`）。
- 业务层**禁止直写 SQL**：账本查询统一走 `ShortTermLedger` 的封装方法。

## 3. 五条链路

### 3.1 拦放链路（接管官方 LLM）

`main.py`：`record_incoming_message`（priority 100，只记账）→ `sleep_gate`（95，作息）→ `gate_official_llm`（90）。

判定分两层，边界必须清晰：

- **规则层** `engine.rule_decide()`：同步、零 token。上游指令 / 无文本 / 私聊 / 被点名在这里当场定夺，返回 `None` 表示「交给巡检」。
- **巡检层** `patrol.PatrolManager`：群聊普通消息**扣押进窗口**，攒一小段再整体判一次。

私聊是中间态：规则层判定「要回」并置 `Decision.debounce=True`，交给 `main._gate_silent_debounce()`，用**静默窗口**等满 `decision.private_debounce_seconds`（默认 10）才唤醒主脑。静默窗口不问模型、不进冷却，计时基准跟着最新那条消息走，并受 `private_debounce_max_seconds`（默认 60，0 = 不封顶）约束。

群内 @ 不必判定是否在叫她：规则层返回 `Decision(debounce=True)`，与私聊一样进入静默窗口，等安静后直接回复。@ 的判定按消息组件里的 `At.qq == self_id` 精确识别，**排除 @全体 / 引用回复 / 唤醒前缀**。静默窗口期间涌入的普通消息只顺延计时，不会顶掉持有窗口的 @ 事件。文本里**只是出现名字**时返回 `Decision(mention=True)`，交给巡检窗口；判定会带上 `mention=True`，要求模型先区分「在叫她」与「只是在讨论智能体、顺口提到名字」，后者判为不回复。**问她的上下文 / 记忆 / 插件 / 工具 / 刚才说了什么，按「在叫她」处理**——这是在跟她说，不是旁观讨论（写在 `_MENTION_NOTE` 里）。

`engine.judge_window()` 是面向「一段对话」的模型判定，只由巡检器调用：

- 输入 = **人格提示词 + 永久记忆 + 相关召回 + 其它会话最近消息 + 本会话前文 + 窗口增量**。
- 人格来自 `persona_manager.get_default_persona_v3()`（截断 1200 字、缓存 5 分钟），取不到退回「无额外人格设定」，判定照常。
- 前文来自 `ledger.messages_before()`：安静一阵后窗口常常只剩「不懂」一句，没有前文模型只能判「无明确上下文」。
- 判据是「这段话是不是在叫她、有没有非接不可的理由」，**默认不开口**；只有明确的接话点才开口，另一个机器人说的话不作为接话理由。放开或收紧都在 `_WINDOW_PROMPT` 里改。

放行 = `event.is_at_or_wake_command = True`；拦下 = `event.stop_event()` + `_mark_gated()`。

**放行必须经过 `main._release_exclusive()`，不要直接调 `_release()`。** AstrBot 的 LLM 互斥只覆盖同一会话（`session_lock_manager.acquire_lock(event.unified_msg_origin)`），私聊与群聊是两个事件、两个任务，会各自起一轮主脑。统一人格只有一份记忆，两轮并发的现象就是「在群里回答私聊的话题」。因此三条放行路径（`_apply_decision`、`_gate_silent_debounce`、`_gate_by_patrol`）统一走 `_release_exclusive()`：先 `brain.acquire()` 申请占用，拿到才 `_release()`，超时则 `_block()` 并记 INFO「主脑正忙，本条暂不处理」。

占用由 `core/gate/brain.py` 的 `BrainGate` 维护，**没有 `asyncio.Lock`**——检查与置位之间不 `await`，单线程事件循环里就是原子的；真拿锁反而会因占用者异常消失而永久死锁。四收口点谁先到谁释放（重复释放是空操作）：`on_agent_done`（最可靠）、`after_message_sent`、`on_decorating_result`、`stay_silent`。两道兜底时限防「拿到却没释放」：未开始生成 15s（放行后仍可能被白名单 / 会话停用 / 限流拦下，`on_agent_begin` 不会来）、已开始生成 300s；等不到占用的上限是 30s。

判定为**开口**时，巡检把 `reason` 写到事件上（`EXTRA_JUDGE_REASON`），由 `main._inject_judge_thought()` 用 `injector.build_judge_thought_block()` 拼成 `<own_thought>` 追加到本轮请求**末尾**。措辞必须是「你自己刚冒出的念头」：一旦写成「判定模型认为……」，主脑会把它当成另一条消息去回应，出现「主脑和巡检聊天」。这一块只递「为什么值得接」这一条理由，**不写任何判定标准**——该不该接属于人格提示词。主脑认为念头站不住时调 `stay_silent` 收回。

### 3.2 焦点链路（她在哪）

`focus/manager.py` + `session_focus` 表。

- 收消息 / `enter_session` / `say` 过去 → 刷新焦点；安静超时或 `leave_session` / `rest` → 失效。
- 焦点**只约束主动发言**，不约束被动接话——任何消息都能唤醒判定。
- `last_read_at` 是**已读水位线**，与 `focused_at` 分开：`touch()` 只刷新焦点不动它，只有 `enter_session` 与 `say` 成功后 `mark_read()` 推进。它只用来算「离开期间攒下的未读」（`ledger.unread_since()`），收消息、超时都不代表她看过那些消息。

### 3.3 作息链路（她醒着吗）

`schedule/state.py`（纯函数 `plan_window` / `phase_at`）+ `sleep_state` 单行表 + `main.sleep_gate()`。

- 只有 `awake` / `light` / `deep` 三态，**由时间与配置现算**，不存中间态，重启即恢复。
- `deep` 拦下一切（含私聊）；`light` 需累积到 `wake_signal_threshold` 次唤醒信号（私聊或 @）才叫醒；命令消息永远放行。
- 睡眠期记账用 `source='sleep'` + `staged=1`（暂存区）；叫醒/起床时 `release_staged()` 只翻 `staged`，**不动 `source`**。
- 睡眠期收消息**不刷新焦点**；焦点只在叫醒/起床时给叫醒她的那个会话。
- 作息事件提示共用一条通道：`wake_notice_pending=1` 或 `awakened_at > wake_notice_fired_at` 才算待注入，注入后把 `wake_notice_fired_at` 抬到当下完成去重。**起床优先于叫醒**；`wake_notice_fired_at` 是唯一水位线，别再引入第二个「已注入」标记。
- 起床提示统计「本轮睡眠期消息」时，起点必须用 `last_sleep_start()`——起床那一刻 `plan_window()` 已经翻到下一轮。
- 暂存消息（`staged=1`）**不算短期记忆**：`live_entries()` / `live_entry_count()` / 时间线条目查询一律只取 `staged=0`。判断「现在是不是睡眠期」要排除清醒宽限（刚被叫醒的 10 分钟她是醒着的）。
- `format_message_line()` 给 `source='sleep'` 补「（睡觉时收到）」。这是**行格式**，绝不能写进 `content`（时间线剔除当前消息靠 content 精确匹配）。
- 周期 tick 负责跨天重置与阶段转换收口；不要在事件处理器里做重活。

### 3.4 记忆链路

`ledger`（账本）→ `extractor`（`condense` 压缩、`merge` 合并、`build_profile` 画像）→ `store`（位置榜 / 召回 / 晋升）→ `maintenance`（收口 / 降级 / 遗忘 / 晋升 / 合并），`profile` 负责用户画像的存储与整理更新。主脑上下文默认沿用 AstrBot 原生按会话历史，注入拼装走 `injector`，时间线仅在接管模式下由 `timeline` 从账本重建，图片经 `vision` 转述、合并转发经 `forward` 取回，坏图摘除在 `history`。

**主脑上下文默认不接管**：`memory.context_takeover=False`（默认）时 `on_llm_request` **不修改 `req.contexts`**，工具清单、说话人画像、话题相关召回、记忆块、跨会话背景（`<cross_session_timeline>`）与作息提示经 `req.extra_user_content_parts` 以 `TextPart(...).mark_as_temp()` 追加，仅本轮可见、不写持久历史，也不会破坏原生前缀缓存。跨会话背景由 `ledger.timeline_entries_balanced()` 取样：每个会话保留最近 `short_term_context_per_chat` 条、纳入最近 `short_term_context_chats` 个会话，**当前会话始终入选且其 `gated=1` 消息全量保留**（这些消息没进官方历史），活摘要也一并保留；原生历史为空时 `include_current=True` 兜底。开启接管后才由 `timeline.build_timeline()` 把账本（`archived_at = 0 AND condensed_at = 0 AND staged = 0`）加活着的摘要渲染成 `req.contexts`——她说过的话是 `assistant`（只放原话），别人的话是 `user`（带 `[会话标签] 发言人 (绝对时间戳):`），摘要用 `<earlier_summary>` 包裹，末尾标明「你现在在跟谁说话」。此时 AstrBot 的原生按会话历史被整体丢弃（`_rebuild_timeline()`）。数据源只覆盖**今天**，超预算从**最旧的**开始丢。

**用户画像以 QQ 号（`user_id`）为身份**：`profile.py` 的 `ProfileStore` 只负责印象（描述 / 关键词 / 更新节拍），名字由 `user_ledger.user_names` 实时累积——两者分开，改名只更新名字列表，不触发画像重写。**不能用 `(platform, user_id)` 做身份**：`platform` 取自 `unified_msg_origin` 首段（平台实例 ID），同一个 QQ 在群聊与私聊、或两套机器人配置下会落到不同实例上，按平台分行就会长出两份画像；`platform` 只作「首次来自哪里」的来源标记。启动时 `_backfill()` 会补空行并把同 `user_id` 的历史重复行并成一行（发言数累加、时间取极值、保留已有印象的那份），名字读取由 `_names()` 按 `user_id` 跨平台合并。`ProfileMaintainer` 在 `MemoryMaintainer.run()` 之后、短期收口**之前**运行：只处理 `profile_window_hours` 内活跃的人，且自 `profiled_at` 之后新增发言达到 `profile_min_messages` 条才调用模型，单次最多 `profile_max_users_per_run` 人。素材有两路：`ledger.user_messages_since()` 的最近发言，以及 `store.memories_of_user()` 取回的**长期 / 永久记忆**——很多认知早已不在短期账本里（被压缩、被清空），只喂发言会让画像反复从头建立。模型输出**完整的新描述 + 新关键词**（整体替换），因此「删除」就是新列表里不再出现它；同时输出 `absorbed`（内容已被新画像完整覆盖、不再有独立价值的记忆编号），由 `_absorb()` 删除这些记忆，跳过 `manual=1`（明确要求记住的）与本次未提交给模型的条目。失败不写 `profiled_at`，下一轮重试，旧画像保持不变。画像块由 `injector.build_profile_block()` 生成，注入时排在最前，只作背景、不得复述。

**主脑每轮还会做一次话题相关召回**：`main._related_memories()` 用当前消息（为空时退回本会话近期文本）调用 `MemoryStore.recall()`，取 `auto_recall_top_k` 条，使用独立预算 `auto_recall_token_budget`，并排除已在永久段 / 榜前正文 / 索引里的条目。它**不调用 `mark_recalled()`**：自动注入若计数，榜前条目会因反复参与而自我强化。只有主动 `recall`、且 message 通道命中分数 ≥ `recall_min_score` 时才计入召回次数并上浮榜序。

**新记忆的入榜位置按对话时间落**：`ledger.condense_context()` 返回 `covered_at`，`maintenance._write_condensed_memories()` 把它作为 `promoted_at`；`created_at` 仍是真实写入时间，用于遗忘的陈旧度判断。只有当天摘要（`_promote_day_summary`）与合并结果（`_apply_merge`）显式落在当下。把普通提炼结果改回 `now` 会让每晚整理把整张榜刷成「最近整理榜」。永久记忆注入另有独立预算 `permanent_token_budget`，否则永久层增长会静默挤空榜前正文与索引。

**消息行必须用绝对时间戳**：`ledger.format_message_line()` 用 `format_clock()`，不要改回 `format_age()`。相对时间会让同一行的文本随当前时刻变化，导致请求前缀逐轮改变、prompt / KV 缓存全部失效。

**短期记忆只有一个出口：压缩。** 逐批提炼已退役。一次压缩同时产出「1 条摘要 + N 条长期记忆」，两者是同源**兄弟**。

- **白天** `condense_short_term()`：条目总数（消息 + 摘要）超 `condense_threshold` 才把最旧 `condense_batch` 条压一轮，职责是「别让上下文太长」。触发点是记账与发送两条路径（`_schedule_condense_if_due()`，30 秒节流 + 防重入），判据是**条数**而非时间，因此不在 `run_maintenance()` 的小时节拍里。
- **深夜** `collapse_short_term()`：树状收口（100:1 收缩），最后唯一那条是**当天摘要**（`_DAY_SUMMARY_CHARS = 400`），由 `_promote_day_summary()` 写进榜最前。收口是短期→长期的**唯一**通道，也是「睡前所有消息都被处理过一遍」的落点。**只有深睡（`full_reset=True`）才收口**：清醒期收口等于当场失忆。

**位置榜取代权重**：长期记忆按 `promoted_at DESC, id DESC` 排。等级（完整 / 精简 / 只在索引里露开头）由**名次**推导（榜前 60% / 85% 分档），不落库。**榜尾那一档不改正文**——模糊只发生在注入侧，`recall` 仍能捞回完整内容。沉到**榜尾 5%** 且降级够多次、够久没被召回才遗忘。`tier='permanent'` 不在榜上，不降级不遗忘，注入时全量排最前。

**深睡整理**：`main._maybe_curate()` 的判据是 **`phase == DEEP` 且 `last_curation_at < window.sleep_at`**（本轮深睡窗口里还没整理过）。每个 60 秒 tick 都看一次，真正的触发点是进入 DEEP 后的第一个刻度。

- **不要改回「距上次满 N 小时」**：绝对节拍的锚点是上次整理**完成**时刻，启动当晚（22:56 装好、01:30 进深睡只隔 2.5 小时）会被整夜挡掉，锚点还会逐日漂移。
- 守卫只推迟不丢失：正在整理、正在压缩、记忆停用、**清醒宽限（刚被叫醒、正在回话）**都跳过，下一 tick 再来。
- `full_reset=True` 顺序固定：维护（降级 → 遗忘 → 晋升 → 合并）→ 无价值记忆（删除上次标记的 + 标记这一轮的）→ 画像更新 → 短期收口 → 提升当天摘要 → `clear_short_term()` → `purge_expired()`。**收口排在维护之后**，当天摘要最后写入，确定性地落在榜首。
- `full_reset=False` 的兜底（不在深睡且距上次超过 `curation_force_after_hours`）**只做维护**，不收口也不清空。
- **关掉作息（`schedule.enabled=False`）则 tick 不跑，永远不会整理。**

**清空不等于删除**：`archived_at > 0` 与 `condensed_at > 0` 只是从常规查询与上下文排除，数据保留 `archive_retention_days`（默认 90 天）供回查。

### 3.5 行动链路

`main.py` 的 LLM 工具：`list_sessions` / `peek_session` / `enter_session` / `say` / `leave_session` / `rest` / `stay_silent` / `remember` / `recall` / `forget` / `search_history` / `delay_sleep`。

「她像人在 QQ 里走动」由三件工具分工承担：

- `peek_session`：**隔窗看一眼**，不改变她在哪儿。想看别处近况、不打算过去时用。
- `enter_session`：**走过去**，不发言。切换焦点 + 推进已读水位线，把「她不在时攒下的未读」作为**工具结果**交给她（不是上下文块，因此不会被当成刚收到的消息）。走进别的会话会置 `EXTRA_ENTERED_TO`。
- `say`：**在那里说一句**。成功后同样刷新焦点并推进已读水位线，置 `EXTRA_LEFT_TO`；`require_focus_for_say` 的「先看后发」会先给出未读预览。

`on_decorating_result` 据此决定本轮收尾发言的去向：`EXTRA_LEFT_TO`（已经说过话）→ 丢弃，那是对原会话的汇报；`EXTRA_ENTERED_TO`（只是走了进去）→ 投递到她进入的会话（`proactive.redirect_reply_to_target`，默认开）。两者同时存在时以 `EXTRA_LEFT_TO` 为准。

两个易混工具的边界：

- `peek_session` 看**当下**（走 `_LIVE_SQL`）；`search_history` 翻**全部**（含已清空与已压进摘要的行）。后者**故意不走 `_LIVE_SQL`**，只排除睡眠期暂存（`staged = 1`）。默认只查当前会话，跨会话必须显式传 `session`；只搜消息不搜摘要。
- `delay_sleep` 只在**清醒**（`phase == AWAKE`）时生效。推迟量记在 `sleep_state.postponed_hours`，**跨天必须归零**（`_sleep_tick` 跨天分支用 0 重算窗口），否则会一天天累积。

## 4. 核心模块速查

| 文件 | 职责 | 改动时的注意点 |
| --- | --- | --- |
| `main.py` | 事件处理器、LLM 工具、生命周期 | 优先级常量在文件顶部，改动前先读注释 |
| `core/common/db.py` | 建表与增量迁移 | 改表结构同步 `_DDL`、`_COLUMN_MIGRATIONS`、`SCHEMA_VERSION`；**建在新增列上的索引必须放 `_POST_MIGRATION_INDEXES`**（放进 `_DDL` 会让老库报 no such column） |
| `core/common/config.py` | 各域配置数据类 | 新增配置项须同步 `_conf_schema.json` |
| `core/common/event_utils.py` | 事件取值与会话标签 | 取不到内部类型时一律降级，不要让上层抛错 |
| `core/common/model_json.py` | 模型输出 JSON 容错解析 | 决断层与压缩共用，改动同时影响两边 |
| `core/gate/engine.py` | 决断层 | `_WINDOW_PROMPT` 是判定质量的直接来源；人格/记忆注入别删 |
| `core/gate/patrol.py` | 巡检窗口 | 扣押 / 替换 / 冷却 / 噪声 / 最近判定快照都在这里 |
| `core/gate/brain.py` | 主脑占用（同一时刻只许一轮） | 放行前 `acquire()`、四收口点 `release()`；**不要改成 `asyncio.Lock`**（占用者异常消失会永久死锁）；`_stale()` 的两档时限是防「拿到不释放」的唯一兜底 |
| `core/focus/manager.py` | 焦点读写 + 已读水位线 | 只约束主动发言，别在被动接话路径上读它；`touch()` 不推进 `last_read_at`，只有 `mark_read()` 推进 |
| `core/memory/ledger.py` | 账本读写、行格式化、条目查询、压缩落库、清空 | `_LIVE_SQL` 是「她此刻应该记得的」唯一表达，**新增查询必须带上它**；`format_message_line` 供判定，`format_speech` 供时间线 |
| `core/memory/store.py` | 位置榜、召回打分、等级推导、遗忘请求 | 榜序 `_BOARD_ORDER` 与分档常量在顶部；等级一律用 `curation_level(rank, total)` 推导，不要新增「等级」列；`promoted_at` 是入榜位置，`created_at` 是产生时间，两者不可混用 |
| `core/memory/maintenance.py` | 压缩 / 树状收口 / 降级 / 遗忘 / 晋升 / 合并 | `run()` 不含收口与清空，那两步在 main；摘要长度常量在文件顶部 |
| `core/memory/profile.py` | 用户画像的存储与整理更新 | 身份是 `user_id`（QQ 号），`platform` 只是来源标记；名字在 `user_ledger`（按 `user_id` 跨平台合并），画像只存印象；素材含长期 / 永久记忆，被完整吸收的条目会删除（`manual=1` 除外）；输出整体替换；更新必须在短期收口之前 |
| `core/memory/injector.py` | 工具清单 / 当前处境 / 画像 / 相关召回 / 记忆注入块 / 判定念头 / 位置提示 + 作息事件提示 | 只负责「长什么样」，「放哪里」在 main；`_TOOL_GUIDE` 必须与 `main.py` 的工具同步；`build_tool_guide_block(available)` 与 `build_situation_block()` 只写**事实**（本轮有哪些工具、在哪个会话、自己的账号），行为准则留在人格提示词 |
| `core/memory/timeline.py` | 账本条目 → `req.contexts`（仅接管模式） | 默认不参与；摘要包 `<earlier_summary>`；每条都要 `_no_save` |
| `core/memory/history.py` | 坏图摘除 | 只做「确定读不到的图片引用」，别恢复官方历史回填 |
| `core/memory/vision.py` | 图片转文字 | 引用必须当场取出，临时文件会失效 |
| `core/memory/forward.py` | 合并转发转文字 | 只认 `Forward.id`；接口参数名跨适配器不一致（`message_id` / `id`）；内容写 `forward_content`，不要改写 `content` |
| `core/schedule/state.py` | 作息推导与状态读写 | 推导保持纯函数；状态只经 `SleepStateStore` 落库 |
| `web_api.py` | 记忆 / 巡检 / 上下文页面接口 | 只开放记忆正文与用户画像的编辑与删除；删除记忆必须先 `suppress()`，否则下一次压缩会写回；画像手工编辑**不推进 `profiled_at`**（那是素材水位线）；短期账本不提供写接口 |
| `pages/memory/` | 记忆页面 | 单文件 HTML，标题靠 i18n 覆盖；写操作用 `bridge.apiPost` |

## 5. 场景导航

**拦放与判定**

- 调整群聊参与时机 → `gate/engine.rule_decide()` + `gate/patrol.py`；静默窗口 / 冷却 / 噪声都在 `DecisionConfig.patrol_*`。群聊是纯静默防抖：每条新消息重置 `patrol_window_seconds`（默认 10），无人说话才判定，**没有封顶**；冷却结束后不补判，等新消息重新开窗。@ 走 `_gate_silent_debounce()`（必回但防抖）；文本提到名字走巡检并附带「是否真的在叫她」的确认。
- 调整私聊与 @ 节奏 → `DecisionConfig.private_debounce_seconds`（0 = 立即回）与 `private_debounce_max_seconds`（最长等待，0 = 不封顶）。实现是 `patrol.hold(..., silent=True)`。
- 调整开口频率 → **开口本身不写冷却**，但她说完话会把窗口顺延一个 `patrol_window_seconds`（`patrol.note_spoke()`，由 `after_message_sent` 与 `say()` 调用）：一问一答不会断，也不会出现「她刚说完、下一句立刻又被判一次」。冷却只有一种，就是**这一轮最终沉默**之后的 `DecisionConfig.patrol_cooldown_seconds`——巡检判了不回复，或主脑用 `stay_silent` 收回（`patrol.enter_cooldown()`），两者都算。连续对话靠**窗口正文的顺序**表达：`patrol._previous_line()` 若发现窗口第一条之前紧邻的一行是 `assistant`（她说的），就把这一行插到正文最前面，判定看到的是「她说 → 对方回」的完整来回，**不要**再加「你刚刚开口了」之类的提示词。
- 区分技术失败 → `Decision.technical=True`（超时 / 异常 / 解析失败）不进入冷却，只丢弃本次窗口。
- 调整判定质量 → `gate/engine.py` 的 `_WINDOW_PROMPT`。它带**人格**与**永久记忆**，但判据必须是「这段话在内容上指向她」；**别把「分享情绪」「气氛合适」这类无门槛理由写回去**——实测它们会让判定模型每窗必开口（13/13 全开口、冷却永不触发），症状是「群里每隔十几秒被插一句」。前文由 `patrol._history_before()` 用 `context_messages` 补。

**记忆**

- 调整注入顺序 → `main.inject_memory_context()`：工具清单 → 当前处境 → 说话人画像 → 话题相关召回 → 永久记忆 → 榜前长期记忆 →（跨会话近况）→（原生历史 / 时间线）→ 判定念头 → 位置提示 → 作息事件提示。默认走 `_append_temp_parts()` 追加临时块；`context_takeover=True` 时才整体替换 `req.contexts`。
- 调整「她知道自己在哪儿」→ `main._situation_block()` + `injector.build_situation_block()`（`<situation>`）。这里只放**事实**（会话标签与群号、自己的 QQ、图片/转发是转述、本轮有无搜索工具），不放行为准则。判定层同理：`engine.judge_window(self_id=...)` 把 `[At:数字]` 的归属写进 `_WINDOW_PROMPT` 的 `{at_note}`。**这些事实一旦下沉到插件，就不要写回人格提示词**——写回去等于把同一个补丁维护两遍，换机器人就失效。
- 调整主脑自动相关召回 → `main._related_memories()`；条数是 `memory.auto_recall_top_k`（0 = 关闭），独立预算是 `auto_recall_token_budget`，阈值复用 `recall_min_score`。它不改变榜序，只有主动 `recall` 才计数。
- 调整跨会话连续性 → `main._cross_session_block()` + `ledger.timeline_entries_balanced()` + `timeline.build_timeline_block()`；配额是 `memory.short_term_context_*`，预算沿用 `timeline_token_budget` / `timeline_max_messages`。
- 调整时间线（仅接管模式）→ `memory/timeline.py` 的 `build_timeline()` 与 `ledger.timeline_entries()`；预算是 `memory.timeline_token_budget` / `timeline_max_messages`。
- 改账本结构 → `common/db.py` 三处一起改；老库靠 `_COLUMN_MIGRATIONS` 补列。
- 调整压缩质量 → `extractor.py` 的 `_CONDENSE_PROMPT` / `_EXTRACT_RULES`；阈值与批大小是 `memory.condense_threshold` / `condense_batch`。
- 调整压缩时机 → `main._schedule_condense_if_due()`（调用点在记账与发送后）；动手门槛在 `maintenance.condense_short_term()`。
- 调整位置榜分档或遗忘 → `store.py` 顶部常量 + `curation_level()` / `is_tail()`；`maintenance._demote()` / `_forget()`；门槛是 `memory.forget_min_curations` / `forget_stale_days`。入榜位置由 `promoted_at` 决定（普通提炼用 `covered_at`，当天摘要 / 合并用当下）；永久层注入预算是 `permanent_token_budget`。
- 调整「无价值记忆」的判断 → `maintenance.curate_worthless()` + `extractor.judge_worthless()` + `store.worthless_candidates()/mark_worthless()/delete_worthless()`；批大小与 TTL 是 `maintenance.py` 顶部的 `_WORTHLESS_BATCH` / `_WORTHLESS_TTL_HOURS`；判据（正文 / 原话 / 关键词三者都只剩分类词）写在 `_WORTHLESS_PROMPT` 里。
- 调整历史回查 → `main.search_history`（参数夹紧、默认当前会话）+ `ledger.search_history()`。它**绕过 `_LIVE_SQL`**，改动前先想清楚哪些行不该被翻出来。
- 调整用户画像 → `memory/profile.py` 的 `ProfileStore`（身份 / 印象）与 `ProfileMaintainer`（整理时更新，节拍是 `memory.profile_*`）；名字来自 `user_ledger.user_names`，素材来自 `ledger.user_messages_since()` + `store.memories_of_user()`，被画像完整吸收的条目由 `_absorb()` 删除（`manual=1` 除外）；注入在 `main._speaker_profile()` + `injector.build_profile_block()`；页面走 `web_api.profiles`。
- 调整图片转述 → `memory/vision.py`（取图、data URL、缓存、模型能力检查）。
- 调整合并转发 → `memory/forward.py`（取 id、调 `get_forward_msg`、渲染与截断）；条数 / 字数是 `memory.forward_max_items` / `forward_max_chars`。

**作息**

- 调整整理节拍 → `schedule.curation_*` + `main._maybe_curate()` / `_run_curation()`；深睡边界由 `wake_notice_lead_minutes` / `light_until_deep_minutes` 决定。
- 调整清空或保留期 → `ledger.clear_short_term()` / `purge_expired()` + `_run_curation()`；开关是 `schedule.daily_reset_enabled` / `archive_retention_days`。
- 调整作息时间或三态 → `schedule/state.plan_window` / `phase_at`；叫醒与暂存在 `main.sleep_gate()` / `_on_wake_up()`。
- 调整推迟入睡 → `main.delay_sleep` + `sleep_state.postponed_hours` + `state._window_for_day`；**跨天归零在 `_sleep_tick`**。

**行动与页面**

- 新增 LLM 工具 → `main.py` 的 `@filter.llm_tool(name=...)`，docstring 就是给模型看的说明书，要写清「什么时候该用」。**同时更新 `injector._TOOL_GUIDE`**：那份清单每轮注入，是模型看到工具的直接来源，漏改会让它照着一份过期的工具表做事。
- 调整切换会话的体验 → `main.enter_session` / `_render_enter` + `focus.mark_read()` + `ledger.unread_since()`；水位线在 `session_focus.last_read_at`，未读的条数与字数上限是 `main._UNREAD_LIMIT` / `_UNREAD_CHARS`；收尾发言去向在 `on_decorating_result` 与 `proactive.redirect_reply_to_target`，位置提示在 `injector.build_whereabouts_block()`。
- 调整主动发言风控 → `proactive` 配置组 + `main._check_say_gate()` / `_check_rate_limit()`。
- 查看主脑上下文 → `main._capture_context_snapshot()` + `web_api.MemoryPageAPI.context()`；快照只存内存（每会话一份、最多 8 个会话），不落库，跟着 `on_llm_request` 更新。
- 查看巡检与焦点 → `patrol.snapshot()`（窗口 / 判定 / 冷却 / 最近判定）+ `focus.list_all()`（含已离开与已超时）+ `web_api.runtime()`；`_remember()` 只存内存，最多 40 个会话。
- 手工编辑与删除记忆 → `store.update_content()`（正文与关键词一起改）+ `web_api.update_memory`；删除先 `store.suppress()` 再 `store.delete()`，否则下一次压缩会把同一条写回来。
- 手工编辑与删除画像 → `ProfileStore.edit()` / `delete()` + `web_api.update_profile` / `delete_profile`；编辑**不推进 `profiled_at`**（那是素材水位线，推进了这段发言就永久漏读）；删除只删画像行，名字与发言记录留在 `user_ledger`。
- 新增页面 → `pages/<名字>/index.html` + `web_api.py` 注册路由 + i18n 配 `views.<名字>.title`。

## 6. 关键机制与坑

**事件优先级（数值越大越早）**：记账 100 / 作息 95 / 决断 90 / 工具清理 60 / 结果装饰 -100 / 发送后收口 100；上下文注入 -10、坏图清理 -20。

- 决断 90 是「霸道」取值：被拦下的消息其它插件（含天使之心）也看不到。与其他插件冲突时调低，只要仍高于默认 LLM 阶段。
- 作息 95 **必须早于决断 90**，否则睡眠期消息仍会进判定。
- `on_llm_request` 默认**不覆盖** `req.contexts`，只追加临时内容块；开启 `memory.context_takeover` 后会整体替换，此时必须晚于同样覆盖 `req.contexts` 的插件（天使之心 50），否则注入块会被抹掉。

**记账必须快**：记账是事件链第一环，任何耗时操作（模型调用、下载）都要 `self._spawn()` 丢后台。

**同一时刻只允许一轮主脑**：AstrBot 只为同一 `unified_msg_origin` 加 LLM 锁，私聊与群聊并发时两轮会同时生成；统一人格只有一份记忆，于是「在群里回答私聊的话题」。插件用 `core/gate/brain.py` 的 `BrainGate` 自己补这层互斥：放行前 `acquire()`、四收口点 `release()`。几处不要改坏：

- **必须走 `_release_exclusive()`**，直接 `_release()` 会绕过闸门。`_release_exclusive()` 返回 True 表示已放行，False 表示已 `_block()`，调用方不要再补 block。
- **不要换成 `asyncio.Lock`**：占用者若在异常路径上消失，锁会被永久带走。现在的实现是「检查 + 置位之间没有 `await`」，单线程事件循环里本身就是原子的。
- **释放点宁多勿少**：`on_agent_done`（唯一在所有路径上都触发的钩子）、`after_message_sent`、`on_decorating_result`、`stay_silent`。空消息链不触发 `after_message_sent`，所以只挂它一定漏。
- **两道超时是唯一兜底**：拿到占用却没能开始生成（白名单 / 会话停用 / 限流把事件拦在了 `ProcessStage` 之前）走 15s；开始之后异常消失走 300s。等不到占用上限 30s，超时就把这条拦下——消息仍在账本里，同会话的下一条会把它带进上下文，不是真的漏看。

**回复的记账时间跟着触发消息走**：回复是到 `after_message_sent` 才写进账本的，而分段发送可能已经晚于下一条用户消息（第一段发出后，第二段发出前，对方又说了下一句）。若按当下时间记，回复就会排到下一句后面，模型看到的是「那句话 → 答非所问的回复 → 下一句」，于是把下一句当成新话题（表现为「受不了了」之后回「怎么了呀」）。`main._reply_created_at()` 取 `EXTRA_MESSAGE_ID` 那条消息的 `created_at + 1ms`，取不到（主动发言）才用当下时间。不要改回 `time.time()`，也不要为此在排序层面加特例——账本只认 `(created_at, id)`。

**自己的消息不参与判定**：`record_incoming_message` 与 `gate_official_llm` 都用 `_is_self_message()` 提前返回。部分适配器会把机器人发出的消息也回灌成事件；判定若把它当成群友发言，就会开出一个只含 assistant 行的巡检窗口，判为沉默后进入冷却，紧接着的下一句话正好被这段冷却挡掉（表现为「她刚说完，我立刻回一句，她就不理了」）。`patrol._decide()` 另有兜底：窗口里只有 assistant 行时既不当作沉默理由，也不进冷却。

**画像更新必须在短期收口之前**：`ProfileMaintainer` 的素材来自 `ledger.user_messages_since()`（走 `_LIVE_SQL`）与 `store.memories_of_user()`（长期 / 永久记忆）。`_run_curation` 里先维护、再更新画像、最后才 `collapse_short_term()`；把画像放到收口之后就只剩空账本，画像会永远停在旧版本且不报错。画像输出是**整体替换**（描述 + 关键词一起重写），不要改成「与旧关键词取并集」——那样被删除的关键词会被合并回来。模型同时给出 `absorbed`，`_absorb()` 会删除这些已被画像完整覆盖的记忆：只删本次提交过的条目、跳过 `manual=1`；这条删除是**不可恢复**的，不要放宽成「按相似度批量删」。模型失败时不写 `profiled_at`，旧画像保持不变，下一轮重试。

**无价值记忆是两段式，不要合并成一步**：整理时只打 `worthless` 标记，删除要等标记满 `_WORTHLESS_TTL_HOURS`（24 小时）。中间这一整轮是模型误判的纠错窗口——被 `recall` 想起（`mark_recalled`）、被重新提到（`reinforce_or_add`）、被手工编辑（`update_content`）都会清掉标记。只有长期层参与判断与删除：永久层不该因为「判断没价值」而消失，晋升后的条目也不会被 `delete_worthless()` 误删（它只删 `tier='long'`）。`worthless_checked_at` 用于取材轮转，没有它每晚都会拿同一批「判断过但仍有价值」的旧记忆重复问模型。判据是「正文 / 原话 / 关键词**同时**只剩分类词」，写在 `_WORTHLESS_PROMPT`。

**未读是「她不在时发生的事」，不是新收到的消息**：`enter_session` 把未读放**工具返回值**里，而不是 `req.extra_user_content_parts`。工具结果天然是「她看到的」，不会被模型当成新一轮用户发言去逐条回应——这正是「切换会话后不把未读当成刚收到的消息」的落点。同理，未读只走 `ledger.unread_since()`（`_LIVE_SQL` + `staged = 0` + `role='user'`），已归档、已压进摘要、睡眠暂存、她自己的话都不算。未读不参与巡检，也不刷新判定状态。

**收尾发言的去向只有一条规则**：本轮若用 `say` 说过话（`EXTRA_LEFT_TO`），收尾一律丢弃——那是对原会话的汇报，发到目标会话是错位；本轮若只是 `enter_session` 走了进去（`EXTRA_ENTERED_TO`），收尾投递到她进入的会话，并同样写回账本（`gated=1`）。投递前必须先 `result.chain = []`，否则同一段内容会被 AstrBot 发回原会话。限流与 `say` 共用同一份滑动窗口。

**`content` 是唯一事实源，`gated` 只是展示用**：`gated=1` 只用于记忆页面与排障，**不要在上下文组装路径上读它**。但**永远不要改写已入账的 `content`**——巡检判据与时间线剔除当前消息都靠它精确匹配。

**图片描述不在 `content` 里**：账本里图片永远是 `[图片]` 占位，描述存 `image_caption`，由 `format_message_line()` 拼成 `[图片]（图片内容：…）`。

**转发内容也不在 `content` 里**：合并转发在账本里只有 `[转发消息]` 占位，内容存 `forward_content`，由 `format_message_line()` 拼成 `[转发消息]（转发内容：…）`；巡检的噪声预过滤必须把它算进可判定文本，否则「转发了一段聊天」会被当成纯占位。

**巡检是扣押，不是轮询**：`patrol.hold()` 返回 Future（`PROCESS` / `KILL`）或 `None`（冷却中，应立即拦下）。同一会话判定必须串行（`_judging`）；窗口到点时若该会话仍在判定，直接作废本次窗口，不排队重试。判定取样只看**本次窗口开启之后**的消息（`opened_at`），冷却期或缺席期间的消息不补判——错过就错过，只有新消息才重新开窗。冷却只有一种语义：「这一轮最终沉默」之后**完全关掉**该会话，期间不调用模型、不处理积压。来源有两个：巡检判了不回复（`_timer` 里写 `_cooldown_until`），或主脑用 `stay_silent` 收回（`patrol.enter_cooldown()`，它还会把正在等的窗口一起 KILL——那批消息属于同一轮）。**@、私聊与文本提到名字都能穿透冷却**（`hold()` 的守卫是 `if not silent and not mention and existing is None`），且一旦这样就开了新窗口，残留的冷却当场清掉——否则会出现「叫了她、她答了，紧接着的闲聊又被旧冷却挡住」；穿透仍不调用模型以外的额外判定。**开口不写冷却**，但 `patrol.note_spoke()`（`after_message_sent` 与 `say` 成功后调用）会把她说完话之后仍在等的那次计时从此刻重新起算（只延时间，不动 `opened_at`）：一问一答不断，她也不会刚说完就被下一句贴身再判一次，而她说话期间进来的消息仍会被判到。连续对话不靠提示词：`patrol._previous_line()` 把窗口第一条之前紧邻的她的那句话插进正文最前面，判定看到的是一段按顺序排列的来回（`你 (21:09:54): …` → `老师 (21:10:23): …`），**别再往 `_WINDOW_PROMPT` 里加「你刚刚开口了」之类的说明**。未调用模型就跳过（噪声 / 空窗口）的日志只记 DEBUG。

**`_LIVE_SQL` 漏加 = 「她已经忘了」静默失效**：`archived_at = 0 AND condensed_at = 0` 少一个条件，已清空或已压进摘要的行就会重新出现在上下文、巡检、压缩与页面里，而且不报错。新增查询时先问：「这条她现在还该记得吗？」

**没收口完就不许清空短期层**：`_run_curation(full_reset=True)` 里 `collapse.failed`、未配置压缩模型、`daily_reset_enabled=False` 三种情况一律跳过清空并告警。兜底整理连收口都不做——清醒期收口等于当场失忆。

**摘要不豁免清空**：「整理完短期层为空」是验收点：当天摘要已先提升为长期记忆，其余摘要与原始消息一起收口。

**名次的唯一事实源是查询填的 `rank` / `rank_total`**：等级由 `curation_level()` 推导、不入库。页面、降级、遗忘都必须走 `store.board()` / `ranked()` / `list_page()`，它们才会填名次。

**人格说明在 `system_prompt`，人格预设对话在 `contexts` 最前**：本插件从不写 `req.system_prompt`；但 AstrBot 会把 `begin_dialogs` 用 `req.contexts[:0] = ...` 插到最前（每条带 `_no_save`），随后才轮到我们整体替换 `req.contexts`。所以必须先用 `_persona_dialogs()` 留下头部连续的 `_no_save` 段，否则人格示例会被静默抹掉。

**插件页面必须显式引入 bridge**：`pages/*/index.html` 的 `<head>` 要写 `<script src="/api/plugin/page/bridge-sdk.js"></script>`。若依赖宿主注入，它会晚于页面内联脚本执行，而插件页 iframe 是 sandbox 的 opaque origin，直连必然 401。同理**不要用 `window.confirm` / `alert` / `prompt`**：sandbox 没有 `allow-modals` 时浏览器会静默忽略弹窗并返回 `false`，表现为「点了没反应」——二次确认要在页面里就地做。

**生命周期**：耗时 `await`（尤其等模型）之后检查 `self._is_terminating`；插件卸载后数据库连接会关，`db._NullCursor` 只负责让迟到的读写静默失效，不要依赖它做正确性保证。

## 7. 测试与验证

**先探查部署环境**：`docker ps` 确认存在 `astrbot` 容器；存在时必须用容器验证，宿主机环境与运行环境不符。

```bash
# 编译与配置校验
docker exec astrbot sh -lc "cd /AstrBot/data/plugins/astrbot_plugin_shittim && python -m compileall -q ."

# 冒烟测试：复制到容器 /tmp 再跑，不要污染运行中的插件目录
docker exec astrbot rm -rf /tmp/astrbot_plugin_shittim
docker cp . astrbot:/tmp/astrbot_plugin_shittim
docker exec astrbot sh -lc 'cd /AstrBot && PYTHONPATH=/tmp:/AstrBot python /tmp/xxx_smoke.py'
```

- 读真实库时**先复制副本**（如 `/tmp/memtest/`）再打开，别对运行中的库做写入测试。
- `main.py` 能否导入也要验：`python -c "import astrbot_plugin_shittim.main"`，编译通过不等于导入通过。
- 改完 `_conf_schema.json` 与 i18n 后，用 `json.load` 校验一次。

## 8. 提交、版本与市场合规

**版本**

- 只在 `metadata.yaml` 的 `version` 维护，不写进 `@register`。
- 用**裸 semver**（如 `1.5.0`），不要 `v` 前缀——市场规范要求 `metadata.yaml.version` 与市场记录逐字符一致。
- 影响运行行为的改动，提交前至少递增末位。

**市场元数据**（[规范 2026-06-27](https://docs.astrbot.app/dev/plugin-market/2026-06-27.html)）

- `plugin_id = author/name = SouthThink/astrbot_plugin_shittim`；市场记录的根 key 必须与之一致。
- 包内 `metadata.yaml` 必填 `name` / `desc` / `version` / `author`；市场记录另需 `repo`，且 `author` / `name` / `version` / `repo` 必须与本包完全一致（`repo` 固定为 `https://github.com/SouthThink/astrbot_plugin_shittim`）。
- `metadata.author` / `name` / `version` 必须与市场记录完全相同；安装后身份校验失败会直接失败。
- 可使用 `display_name` / `short_desc` / `logo` / `tags` / `support_platforms` / `astrbot_version`；**不得**出现 `plugin_id`、`root_dir_name`、`install_method` 等保留字段。
- 仓库里的 `build/` 不进版本库（已 gitignore）。

**提交信息**：`type(scope): 中文摘要`，常用 `feat` / `fix` / `refactor` / `perf` / `docs` / `chore`。

## 9. 文档职责

- `README.md` 面向使用者：只写能力、用法、配置与安装，**不重复目录树、数据表等仓库里一眼可见的实现细节**。
- `AGENTS.md`（本文件）面向开发者：架构边界、坑、测试与发布规则。
- AstrBot 的开发资料在 `docs/`；涉及事件注入、提示词注入、主动消息、请求体组建时先查那里。

## 10. 配置页描述规范

`_conf_schema.json` 的 `description` 是**面向使用者的唯一说明**，必须能脱离源码独立读懂。新增或修改配置项时逐条检查：

**内容**

- 一句话讲清「打开会发生什么 / 关闭会发生什么」，不要只把键名翻译一遍。
- 写全**单位与取值范围**：秒 / 分钟 / 小时 / 天 / 条 / token / 1~10 / 0~1，单位放在括号里。
- 写清**特殊值语义**：`0` 表示关闭、不封顶还是立即返回；留空表示复用全局配置还是禁用功能。
- 写清**作用域与消费者**：同一概念出现在多处时，必须点明是「判定上下文」「主脑背景」「接管时间线」中的哪一个，避免读者以为它们共用一份预算。
- 层级开关要说明关系：总开关关闭后哪些子功能一并失效，子开关只影响哪一步。
- **副作用必须写明**：例如关闭作息不仅是不限制在线时段，也不会自动整理记忆。
- 固定格式的配置（白名单、别名）给出示例，如 `private:QQ号` / `group:群号`。

**措辞**

- 使用陈述句，不写「搞一下」「弄成」「可能会」这类口语或模糊表达。
- 不用「强度值」「一些」「若干」这类没有刻度的词；能给出区间就给出区间。
- 不写内部实现细节（函数名、表名、SQL、变量名），只写使用者能观察到的结果。
- 全仓库统一术语：跨会话背景、判定上下文、静默防抖、接管上下文、位置榜。
- 描述控制在 1~2 句；较长解释放 README，不堆进配置项。

**维护**

- 新增或修改配置项时，同步 `_conf_schema.json`、`core/common/config.py` 的数据类与 `from_dict()`，三处默认值必须一致。
- 已发布的键名不因描述修正而改名；键名与语义不符时（如 `private_debounce_seconds` 现在也管 @），保留键名并在描述中写清实际范围。
- 改完执行 `python -c "import json; json.load(open('_conf_schema.json', encoding='utf-8'))"` 校验。

**示例**

- ✗「是否启用记忆维护」：读者不知道维护包含什么，也不知道它和深睡整理的关系。
- ✓「记忆维护（降级 / 遗忘 / 晋升 / 合并）开关；不影响短期压缩与深睡收口」
- ✗「跨会话语境的 token 预算」：三处跨会话内容都在用，读者无法判断改它影响谁。
- ✓「判定用跨会话语境的 token 预算（决断层与巡检；主脑背景用 timeline_token_budget）」
