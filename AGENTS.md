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
├── memory/     记忆链路：ledger / extractor / store / maintenance
│               / timeline / injector / history / vision
└── schedule/   作息链路：state.py

main.py         事件处理器、LLM 工具、生命周期
web_api.py      记忆页面只读接口
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

私聊是中间态：规则层只定「要回」，`Decision.debounce=True` 交给 `main._gate_private_debounce()`，用**静默窗口**等满 `decision.private_debounce_seconds`（默认 10）才唤醒主脑。静默窗口不问模型、不进冷却、不设封顶，计时基准跟着最新那条消息走。

`engine.judge_window()` 是面向「一段对话」的模型判定，只由巡检器调用：

- 输入 = **人格提示词 + 永久记忆 + 相关召回 + 其它会话最近消息 + 本会话前文 + 窗口增量**。
- 人格来自 `persona_manager.get_default_persona_v3()`（截断 1200 字、缓存 5 分钟），取不到退回「无额外人格设定」，判定照常。
- 前文来自 `ledger.messages_before()`：安静一阵后窗口常常只剩「不懂」一句，没有前文模型只能判「无明确上下文」。
- 判据是「这段话里有没有她想接的东西」，**默认倾向参与**而不是沉默；沉默只留给纯刷屏、别人的私事、刚说过的话题。

放行 = `event.is_at_or_wake_command = True`；拦下 = `event.stop_event()` + `_mark_gated()`。

### 3.2 焦点链路（她在哪）

`focus/manager.py` + `session_focus` 表。

- 收消息 / `say` 过去 → 刷新焦点；安静超时或 `leave_session` / `rest` → 失效。
- 焦点**只约束主动发言**，不约束被动接话——任何消息都能唤醒判定。

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

`ledger`（账本）→ `extractor`（`condense` 压缩、`merge` 合并）→ `store`（位置榜 / 召回 / 晋升）→ `maintenance`（收口 / 降级 / 遗忘 / 晋升 / 合并）。主脑上下文由 `timeline` 从账本整体重建，注入拼装走 `injector`，图片经 `vision` 转述，坏图摘除在 `history`。

**全局时间线是主脑唯一的上下文**：`timeline.build_timeline()` 把账本（`archived_at = 0 AND condensed_at = 0 AND staged = 0`）加活着的摘要渲染成 `req.contexts`——她说过的话是 `assistant`（只放原话），别人的话是 `user`（带 `[会话标签] 发言人 (多久前):`），摘要用 `<earlier_summary>` 包裹，末尾标明「你现在在跟谁说话」。所以**切会话不换上下文**，只有最后那条消息换了人。AstrBot 的原生按会话历史被整体丢弃（`_rebuild_timeline()`）。数据源只覆盖**今天**，超预算从**最旧的**开始丢。

**短期记忆只有一个出口：压缩。** 逐批提炼已退役。一次压缩同时产出「1 条摘要 + N 条长期记忆」，两者是同源**兄弟**。

- **白天** `condense_short_term()`：条目总数（消息 + 摘要）超 `condense_threshold` 才把最旧 `condense_batch` 条压一轮，职责是「别让上下文太长」。触发点是记账与发送两条路径（`_schedule_condense_if_due()`，30 秒节流 + 防重入），判据是**条数**而非时间，因此不在 `run_maintenance()` 的小时节拍里。
- **深夜** `collapse_short_term()`：树状收口（100:1 收缩），最后唯一那条是**当天摘要**（`_DAY_SUMMARY_CHARS = 400`），由 `_promote_day_summary()` 写进榜最前。收口是短期→长期的**唯一**通道，也是「睡前所有消息都被处理过一遍」的落点。**只有深睡（`full_reset=True`）才收口**：清醒期收口等于当场失忆。

**位置榜取代权重**：长期记忆按 `promoted_at DESC, id DESC` 排。等级（完整 / 精简 / 关键词）由**名次**推导（榜前 60% / 85% 分档），不落库。沉到**榜尾 5%** 且降级够多次、够久没被召回才遗忘。`tier='permanent'` 不在榜上，不降级不遗忘，注入时全量排最前。

**深睡整理**：`main._maybe_curate()` 的判据是 **`phase == DEEP` 且 `last_curation_at < window.sleep_at`**（本轮深睡窗口里还没整理过）。每个 60 秒 tick 都看一次，真正的触发点是进入 DEEP 后的第一个刻度。

- **不要改回「距上次满 N 小时」**：绝对节拍的锚点是上次整理**完成**时刻，启动当晚（22:56 装好、01:30 进深睡只隔 2.5 小时）会被整夜挡掉，锚点还会逐日漂移。
- 守卫只推迟不丢失：正在整理、正在压缩、记忆停用、**清醒宽限（刚被叫醒、正在回话）**都跳过，下一 tick 再来。
- `full_reset=True` 顺序固定：维护（降级 → 遗忘 → 晋升 → 合并）→ 短期收口 → 提升当天摘要 → `clear_short_term()` → `purge_expired()`。**收口排在维护之后**，当天摘要最后写入，确定性地落在榜首。
- `full_reset=False` 的兜底（不在深睡且距上次超过 `curation_force_after_hours`）**只做维护**，不收口也不清空。
- **关掉作息（`schedule.enabled=False`）则 tick 不跑，永远不会整理。**

**清空不等于删除**：`archived_at > 0` 与 `condensed_at > 0` 只是从常规查询与上下文排除，数据保留 `archive_retention_days`（默认 90 天）供回查。

### 3.5 行动链路

`main.py` 的 LLM 工具：`list_sessions` / `peek_session` / `say` / `leave_session` / `rest` / `remember` / `recall` / `forget` / `search_history` / `delay_sleep`。

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
| `core/gate/patrol.py` | 巡检窗口 | 扣押 / 替换 / 冷却 / 噪声四件事都在这里 |
| `core/focus/manager.py` | 焦点读写 | 只约束主动发言，别在被动接话路径上读它 |
| `core/memory/ledger.py` | 账本读写、行格式化、条目查询、压缩落库、清空 | `_LIVE_SQL` 是「她此刻应该记得的」唯一表达，**新增查询必须带上它**；`format_message_line` 供判定，`format_speech` 供时间线 |
| `core/memory/store.py` | 位置榜、召回打分、等级推导、遗忘请求 | 榜序 `_BOARD_ORDER` 与分档常量在顶部；等级一律用 `curation_level(rank, total)` 推导，不要新增「等级」列 |
| `core/memory/maintenance.py` | 压缩 / 树状收口 / 降级 / 遗忘 / 晋升 / 合并 | `run()` 不含收口与清空，那两步在 main；摘要长度常量在文件顶部 |
| `core/memory/injector.py` | 注入块拼装 + 作息事件提示 | 只负责「长什么样」，「放哪里」在 main |
| `core/memory/timeline.py` | 账本条目 → `req.contexts` | 主脑唯一上下文；摘要包 `<earlier_summary>`；每条都要 `_no_save` |
| `core/memory/history.py` | 坏图摘除 | 只做「确定读不到的图片引用」，别恢复官方历史回填 |
| `core/memory/vision.py` | 图片转文字 | 引用必须当场取出，临时文件会失效 |
| `core/schedule/state.py` | 作息推导与状态读写 | 推导保持纯函数；状态只经 `SleepStateStore` 落库 |
| `web_api.py` | 记忆页面只读接口 | 路由注册在 `main.__init__` |
| `pages/memory/` | 记忆页面 | 单文件 HTML，标题靠 i18n 覆盖 |

## 5. 场景导航

**拦放与判定**

- 调整群聊参与时机 → `gate/engine.rule_decide()` + `gate/patrol.py`；窗口 / 封顶 / 冷却 / 噪声都在 `DecisionConfig.patrol_*`。
- 调整私聊节奏 → `DecisionConfig.private_debounce_seconds`（0 = 立即回）。实现是 `patrol.hold(..., silent=True)`。
- 调整判定质量 → `gate/engine.py` 的 `_WINDOW_PROMPT`。它带**人格**与**永久记忆**，倾向参与而非沉默，别把「宁可不开口」写回去；前文由 `patrol._history_before()` 用 `context_messages` 补。

**记忆**

- 调整注入顺序 → `main.inject_memory_context()`：永久记忆 → 长期记忆 →（全局时间线）→ 当前消息 → 作息事件提示。前两段在时间线**之前**，作息提示追加在**最新处**。
- 调整时间线 → `memory/timeline.py` 与 `ledger.timeline_entries()`；预算是 `memory.timeline_token_budget` / `timeline_max_messages`。
- 改账本结构 → `common/db.py` 三处一起改；老库靠 `_COLUMN_MIGRATIONS` 补列。
- 调整压缩质量 → `extractor.py` 的 `_CONDENSE_PROMPT` / `_EXTRACT_RULES`；阈值与批大小是 `memory.condense_threshold` / `condense_batch`。
- 调整压缩时机 → `main._schedule_condense_if_due()`（调用点在记账与发送后）；动手门槛在 `maintenance.condense_short_term()`。
- 调整位置榜分档或遗忘 → `store.py` 顶部常量 + `curation_level()` / `is_tail()`；`maintenance._demote()` / `_forget()`；门槛是 `memory.forget_min_curations` / `forget_stale_days`。
- 调整历史回查 → `main.search_history`（参数夹紧、默认当前会话）+ `ledger.search_history()`。它**绕过 `_LIVE_SQL`**，改动前先想清楚哪些行不该被翻出来。
- 调整图片转述 → `memory/vision.py`（取图、data URL、缓存、模型能力检查）。

**作息**

- 调整整理节拍 → `schedule.curation_*` + `main._maybe_curate()` / `_run_curation()`；深睡边界由 `wake_notice_lead_minutes` / `light_until_deep_minutes` 决定。
- 调整清空或保留期 → `ledger.clear_short_term()` / `purge_expired()` + `_run_curation()`；开关是 `schedule.daily_reset_enabled` / `archive_retention_days`。
- 调整作息时间或三态 → `schedule/state.plan_window` / `phase_at`；叫醒与暂存在 `main.sleep_gate()` / `_on_wake_up()`。
- 调整推迟入睡 → `main.delay_sleep` + `sleep_state.postponed_hours` + `state._window_for_day`；**跨天归零在 `_sleep_tick`**。

**行动与页面**

- 新增 LLM 工具 → `main.py` 的 `@filter.llm_tool(name=...)`，docstring 就是给模型看的说明书，要写清「什么时候该用」。
- 调整主动发言风控 → `proactive` 配置组 + `main._check_say_gate()` / `_check_rate_limit()`。
- 新增页面 → `pages/<名字>/index.html` + `web_api.py` 注册路由 + i18n 配 `views.<名字>.title`。

## 6. 关键机制与坑

**事件优先级（数值越大越早）**：记账 100 / 作息 95 / 决断 90 / 工具清理 60 / 结果装饰 -100 / 发送后收口 100；上下文注入 -10、坏图清理 -20。

- 决断 90 是「霸道」取值：被拦下的消息其它插件（含天使之心）也看不到。与其他插件冲突时调低，只要仍高于默认 LLM 阶段。
- 作息 95 **必须早于决断 90**，否则睡眠期消息仍会进判定。
- `on_llm_request` 必须晚于会整体覆盖 `req.contexts` 的插件（天使之心 50），否则注入块会被抹掉。

**记账必须快**：记账是事件链第一环，任何耗时操作（模型调用、下载）都要 `self._spawn()` 丢后台。

**`content` 是唯一事实源，`gated` 只是展示用**：`gated=1` 只用于记忆页面与排障，**不要在上下文组装路径上读它**。但**永远不要改写已入账的 `content`**——巡检判据与时间线剔除当前消息都靠它精确匹配。

**图片描述不在 `content` 里**：账本里图片永远是 `[图片]` 占位，描述存 `image_caption`，由 `format_message_line()` 拼成 `[图片]（图片内容：…）`。

**巡检是扣押，不是轮询**：`patrol.hold()` 返回 Future（`PROCESS` / `KILL`）或 `None`（冷却中，应立即拦下）。同一会话判定必须串行（`_judging`），窗口起点用 `_last_judged_at` 保证冷却期间的对话不丢。

**`_LIVE_SQL` 漏加 = 「她已经忘了」静默失效**：`archived_at = 0 AND condensed_at = 0` 少一个条件，已清空或已压进摘要的行就会重新出现在上下文、巡检、压缩与页面里，而且不报错。新增查询时先问：「这条她现在还该记得吗？」

**没收口完就不许清空短期层**：`_run_curation(full_reset=True)` 里 `collapse.failed`、未配置压缩模型、`daily_reset_enabled=False` 三种情况一律跳过清空并告警。兜底整理连收口都不做——清醒期收口等于当场失忆。

**摘要不豁免清空**：「整理完短期层为空」是验收点：当天摘要已先提升为长期记忆，其余摘要与原始消息一起收口。

**名次的唯一事实源是查询填的 `rank` / `rank_total`**：等级由 `curation_level()` 推导、不入库。页面、降级、遗忘都必须走 `store.board()` / `ranked()` / `list_page()`，它们才会填名次。

**人格说明在 `system_prompt`，人格预设对话在 `contexts` 最前**：本插件从不写 `req.system_prompt`；但 AstrBot 会把 `begin_dialogs` 用 `req.contexts[:0] = ...` 插到最前（每条带 `_no_save`），随后才轮到我们整体替换 `req.contexts`。所以必须先用 `_persona_dialogs()` 留下头部连续的 `_no_save` 段，否则人格示例会被静默抹掉。

**插件页面必须显式引入 bridge**：`pages/*/index.html` 的 `<head>` 要写 `<script src="/api/plugin/page/bridge-sdk.js"></script>`。若依赖宿主注入，它会晚于页面内联脚本执行，而插件页 iframe 是 sandbox 的 opaque origin，直连必然 401。

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
