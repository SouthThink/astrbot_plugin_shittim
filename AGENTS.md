# 什亭之枢 AstrBot 插件开发指引

## 插件定位

什亭之枢（Shittim）让 AstrBot 智能体拥有统一人格、跨会话记忆与主动行动能力：
所有私聊与群聊共享同一份记忆，她能自主决定**对谁说、在哪个会话说、什么时候说**。

一句话区分职责：**决断层决定「要不要说」，主脑决定「怎么说」，插件决定「说在哪里」。**

作息状态机、深睡期记忆整理（短期收口 / 名次降级 / 阶梯遗忘 / 清空账本）已经落地，见下方「作息链路」与「记忆链路」；`plan/` 里只剩 P6（`search_history` 回查工具）与 P7（`delay_sleep` 工具、记忆页面展示等级）未做，不要提前替这两项写代码。

## 目录结构

代码按**功能域**分包，目录名与「五条链路」一一对应，看文档即知去哪找：

```
core/
├── common/     通用基础设施：配置、数据库、事件取用、时间与分词、模型 JSON 解析
├── gate/       拦放链路：决断层 engine.py + 群聊巡检 patrol.py
├── focus/      焦点链路：manager.py
├── memory/     记忆链路：ledger(账本) / extractor(压缩与合并) / store(位置榜) / maintenance(维护)
│               / timeline(全局时间线) / injector(注入块) / history(坏图摘除) / vision(图片转述)
└── schedule/   作息链路：state.py
```

`core/common/` **不依赖任何业务包**，只被它们依赖。新增模块时先判断它属于哪条链路；
跨包引用一律用相对导入（`from ..common.db import MemoryDB`）。

## 架构：五条链路

### 拦放链路（接管官方 LLM）

`main.py`：`record_incoming_message`（priority 100，只记账）→ `sleep_gate`（priority 95，作息）→ `gate_official_llm`（priority 90）。

判定分两层，边界必须保持清晰：

- `core/gate/engine.py: rule_decide()` —— 同步、零 token。上游指令 / 私聊 / 被点名 / 无文本在这里当场定夺，返回 `None` 表示「交给巡检」。
- `core/gate/patrol.py: PatrolManager` —— 群聊里的普通消息**扣押进窗口**，攒一小段再整体判一次。
- 私聊是个中间态：规则层定的只是「要回」，`Decision.debounce=True` 把它交给 `main.py: _gate_private_debounce()`，用巡检器的**静默窗口**攒够 `decision.private_debounce_seconds` 秒（默认 10）才唤醒主脑。静默窗口不问模型、不进冷却、不设封顶，计时基准跟着最新那条消息走。
- `core/gate/engine.py: judge_window()` —— 面向「一段对话」的模型判定，调用方是巡检器，不直接被事件处理器调用。

放行 = `event.is_at_or_wake_command = True`；拦下 = `event.stop_event()` + `_mark_gated()`。

### 焦点链路（她在哪）

`core/focus/manager.py`，`session_focus` 表。焦点**只约束主动发言**，不约束被动接话。
收消息 / `say` 过去 → 刷新焦点；安静超时或 `leave_session` / `rest` → 失效。

### 作息链路（她醒着吗）

`core/schedule/state.py`（纯函数推导 `plan_window` / `phase_at`）+ `sleep_state` 单行表 + `main.py: sleep_gate()`。

- 状态只有 `awake` / `light` / `deep` 三种，**由时间与配置现算**，不存中间态，重启即可恢复。
- `deep` 拦下一切（含私聊）；`light` 只有累积到 `wake_signal_threshold` 次唤醒信号（私聊或 @）才叫醒；命令消息永远放行。
- 睡眠期间记账用 `source='sleep'` + `staged=1`（暂存区）；叫醒/起床时 `release_staged()` 只翻 `staged`，**不动 `source`**。
- 周期性 tick 负责跨天重置与阶段转换收口；不要在事件处理器里做重活。
- 睡眠期收消息**不刷新焦点**，否则睡着的人会靠别人的私聊把自己点亮；焦点只在叫醒/起床时给叫醒她的那个会话。
- 作息事件提示（被叫醒 / 刚起床）共用一条通道：`wake_notice_pending=1`、或 `awakened_at > wake_notice_fired_at`，才算有提示待注入；注入后把 `wake_notice_fired_at` 抬到当下即完成去重。**起床优先于叫醒**。改这部分时注意 `wake_notice_fired_at` 是唯一的水位线，别再引入第二个「已注入」标记。
- 起床提示要算「本轮睡眠期消息」，起点必须用 `last_sleep_start()` 而不是 `plan_window()` —— 起床那一刻窗口已经翻到下一轮了。
- 暂存区的消息（`staged=1`）**不算短期记忆**：`live_entries()` / `live_entry_count()` / 时间线条目查询一律只取 `staged=0`——她那时没在听，这些消息在她醒来前都不该出现。起床放行后它们才入账、才参与当晚收口。判断「现在是不是睡眠期」要排除宽限期——刚被叫醒那 10 分钟她是醒着的，那时收到的消息算实时消息。
- `format_message_line()` 会给 `source='sleep'` 的行补「（睡觉时收到）」标记。这是**行格式**，绝不能写进 `content`（时间线剔除当前消息靠 content 精确匹配）。

### 记忆链路

`memory/ledger.py`（账本）→ `memory/extractor.py`（`condense` 压缩、`merge` 合并）→ `memory/store.py`（位置榜 / 召回 / 晋升）→ `memory/maintenance.py`（收口 / 降级 / 遗忘 / 晋升 / 合并）。主脑的上下文由 `memory/timeline.py` 从账本整体重建，注入拼装走 `memory/injector.py`，图片消息额外走 `memory/vision.py` 转述，坏图摘除在 `memory/history.py`。

**全局时间线是主脑唯一的上下文**：`memory/timeline.py: build_timeline()` 把账本（`archived_at = 0 AND condensed_at = 0 AND staged = 0`）加活着的摘要渲染成 req.contexts——她说过的话是 `assistant` 回合（只放原话），别人的话是 `user` 回合（正文带 `[会话标签] 发言人 (多久前):`），摘要用 `<earlier_summary>` 包裹，末尾一行标明「你现在在跟谁说话」。因此**切会话不换上下文**，只有最后那条消息换了个人。AstrBot 按会话隔离的原生历史被整体丢弃（`main.py: _rebuild_timeline()`），不再叠加也不再合并。数据源天然只覆盖**今天**（清空即重来），超预算从**最旧的**开始丢。

**短期记忆只有一个出口：压缩。** 逐批提炼已退役——同一批原文既提炼又压缩等于处理两遍。压缩一次调用同时产出「1 条摘要 + N 条长期记忆」，两者是同源**兄弟**，不是父子。

- **白天**：`condense_short_term()`，条目总数（消息 + 摘要）超 `condense_threshold` 才把最旧 `condense_batch` 条压一轮。它的职责是「别让上下文太长」，不是清空。
- **深夜**：`collapse_short_term()` 树状收口（每轮分块各压一条，100:1 收缩），最后唯一那条就是**当天摘要**（`_DAY_SUMMARY_CHARS = 400`），由 `_promote_day_summary()` 作为一条长期记忆写进榜最前。收口是短期→长期**唯一**通道，也是「睡前所有消息都被处理过一遍」的落点。

**位置榜取代权重**：长期记忆只按 `promoted_at DESC, id DESC` 排（每晚整理产出、每次召回都刷新为当下）。等级（完整 / 精简 / 关键词）由**名次**推导（榜前 60% / 85% 分档），不落库。沉到**榜尾 5%** 且降级够多次、够久没被召回，才被遗忘。`tier='permanent'` 不在榜上，不降级不遗忘，注入时全量排在最前。

**深睡整理**：`main._run_curation()` 顺序固定——维护（降级 → 遗忘 → 晋升 → 合并）→ 短期收口 → 提升当天摘要 → `clear_short_term()` 清空 → `purge_expired()`。**收口排在维护之后**，当天摘要最后写入（同刻并列靠 id 兜底），于是确定性地落在榜首。

**清空不等于删除**：`archived_at > 0`（原始消息）与 `condensed_at > 0`（已被压进摘要的消息、被吸收的旧摘要）都只是从常规查询与上下文里排除，数据保留 `archive_retention_days`（默认 90 天）供回查。

### 行动链路

`main.py` 的 LLM 工具：`list_sessions` / `peek_session` / `say` / `leave_session` / `rest` / `remember` / `recall` / `forget`。

## 核心模块速查

| 文件 | 职责 | 改动时的注意点 |
| --- | --- | --- |
| `main.py` | 事件处理器、LLM 工具、生命周期 | 优先级常量在文件顶部，改动前先读注释 |
| `core/common/db.py` | 建表与增量迁移 | 改表结构必须同步 `_DDL`、`_COLUMN_MIGRATIONS`、`SCHEMA_VERSION`；**建在新增列上的索引必须放 `_POST_MIGRATION_INDEXES`**（放进 `_DDL` 会让老库报 no such column） |
| `core/common/config.py` | 配置数据类（各域的集中入口） | 新增配置项须同步 `_conf_schema.json` |
| `core/common/event_utils.py` | 事件取值与会话标签 | 取不到内部类型时一律降级，不要让上层抛错 |
| `core/common/model_json.py` | 模型输出的 JSON 容错解析 | 决断层与压缩共用，改动会同时影响两边 |
| `core/gate/engine.py` | 决断层 | `_WINDOW_PROMPT` 是判定质量的直接来源 |
| `core/gate/patrol.py` | 巡检窗口 | 扣押/替换/冷却/噪声四件事都在这里 |
| `core/focus/manager.py` | 焦点读写 | 只约束主动发言，别在被动接话路径上读它 |
| `core/memory/ledger.py` | 账本读写与行格式化、条目查询（含摘要）、压缩落库、清空 | `format_message_line` 是判定/语境块的输入格式，`format_speech` 是时间线里 `assistant` 回合的干净正文；`_LIVE_SQL` 是「她此刻应该记得的」唯一表达；**新增查询必须带上它** |
| `core/memory/store.py` | 位置榜读写、召回打分、等级推导、遗忘请求 | 榜序 `_BOARD_ORDER` 与分档常量（`DEGRADE_*_RATIO`）在文件顶部；等级一律用 `curation_level(rank, total)` 推导，不要新增「等级」列 |
| `core/memory/maintenance.py` | 维护管线：压缩 / 树状收口 / 降级 / 遗忘 / 晋升 / 合并 | `run()` 不含收口与清空，那两步在 main；摘要长度常量（`_SUMMARY_CHARS` / `_DAY_SUMMARY_CHARS`）在文件顶部 |
| `core/memory/injector.py` | 注入块拼装（永久 / 榜前正文 / 关键词索引 + 作息事件提示） | 只负责「长什么样」，「放哪里」在 main |
| `core/memory/timeline.py` | 全局时间线的渲染（账本条目 → req.contexts） | 主脑唯一的上下文；`user` 行带会话/发言人前缀，`assistant` 行只放原话，摘要包 `<earlier_summary>`；每条都要 `_no_save` |
| `core/memory/history.py` | 坏图摘除（官方历史回填已退役） | 只做「确定读不到的图片引用」的摘除，别在这里恢复回填 |
| `core/memory/vision.py` | 图片转文字 | 引用必须当场取出，临时文件会失效 |
| `core/schedule/state.py` | 作息状态推导与状态读写 | 推导必须保持纯函数（同输入同输出），状态只经 `SleepStateStore` 落库 |
| `web_api.py` | 记忆页面只读接口 | 路由注册在 `main.__init__` 里 |
| `pages/memory/` | 记忆页面 | 单文件 HTML，标题靠 i18n 覆盖 |

## 场景导航

- **调整群聊参与时机** → `core/gate/engine.py: rule_decide()` 与 `core/gate/patrol.py`。窗口/封顶/冷却/噪声都在 `DecisionConfig` 的 `patrol_*`。
- **调整私聊回复节奏** → `DecisionConfig.private_debounce_seconds`（0 = 立即回，默认 10）。实现是 `patrol.hold(..., silent=True)` 的静默窗口，被 KILL 的消息仍在时间线上（账本即上下文）。
- **调整判定质量** → `core/gate/engine.py` 的 `_WINDOW_PROMPT`。记住判据是**一段对话**，不是单条消息。
- **调整注入顺序或内容** → `main.py: inject_memory_context()`。顺序 = 模型读取顺序：永久记忆 → 长期记忆 →（全局时间线）→ 当前消息 → 作息事件提示。前两段是长期背景，插在时间线**之前**（`_inject_memory_blocks()`）；作息提示是时效事件，追加在**最新处**（`_inject_schedule_notice()`）。
- **调整时间线的范围或形态** → `core/memory/timeline.py` 与 `ledger.py: timeline_entries()`；预算/上限是 `memory.timeline_token_budget` / `timeline_max_messages`。
- **改账本结构** → `core/common/db.py` 三处一起改；已有库靠 `_COLUMN_MIGRATIONS` 补列。
- **新增 LLM 工具** → `main.py` 里 `@filter.llm_tool(name=...)`，docstring 就是给模型看的说明书，要写清「什么时候该用」。
- **新增页面** → `pages/<名字>/index.html` + `web_api.py` 注册路由 + `.astrbot-plugin/i18n/zh-CN.json` 里配 `views.<名字>.title`。
- **调整压缩质量与形态** → `core/memory/extractor.py` 的 `_CONDENSE_PROMPT` / `_EXTRACT_RULES`；阈值与批大小是 `memory.condense_threshold` / `condense_batch`，顺带产出上限 `extract_max_items`，开关是 `memory.extract_include_events`，摘要长度常量在 `maintenance.py` 顶部。
- **调整位置榜分档或遗忘** → `core/memory/store.py` 顶部的 `DEGRADE_BRIEF_RATIO` / `DEGRADE_KEYWORD_RATIO` / `_FORGET_TAIL_RATIO` 与 `curation_level()` / `is_tail()`；`core/memory/maintenance.py` 的 `_demote()`（含 `_DEMOTE_INTERVAL_HOURS`、`_BRIEF_MAX_CHARS`、`_MERGE_SAFE_RATIO`）与 `_forget()`。遗忘门槛是 `memory.forget_min_curations` / `forget_stale_days`。
- **调整每日整理节拍** → `schedule` 配置组的 `curation_*` + `main.py` 的 `_maybe_curate()` / `_run_curation()`。
- **调整清空或保留期** → `core/memory/ledger.py` 的 `clear_short_term()` / `purge_expired()` 与 `main.py` 的 `_run_curation()`；开关是 `schedule.daily_reset_enabled` / `archive_retention_days`。
- **调整图片转述** → `core/memory/vision.py`（取图、转 data URL、缓存、模型能力检查）。
- **调整主动发言风控** → `proactive` 配置组 + `main.py` 的 `_check_say_gate()` / `_check_rate_limit()`。
- **调整作息时间或三态行为** → `schedule` 配置组 + `core/schedule/state.py` 的 `plan_window` / `phase_at`；叫醒与暂存放在 `main.py: sleep_gate()` / `_on_wake_up()`。

## 关键机制与坑

**事件优先级（数值越大越早）**：记账 100 / 作息 95 / 决断 90 / 工具清理 60 / 结果装饰 -100 / 发送后收口 100。上下文注入 -10、坏图清理 -20 排在最后：`on_llm_request` 也必须晚于会整体覆盖 `req.contexts` 的插件（天使之心 50），否则注入块会被抹掉。
决断层用 90 是「霸道」取值：被拦下的消息，其它插件（含装在一起的天使之心）也看不到。若与其他插件冲突，调低它（只要仍高于默认 LLM 阶段）。作息 95 同理但只作用于睡眠时段，**必须早于决断 90**，否则睡眠期的消息仍会进判定。

**记账必须快**。记账是事件链第一环，任何耗时操作（模型调用、下载）都要用 `self._spawn()` 丢后台，否则会把后面所有插件的处理一起拖慢。

**`content` 是唯一的事实源，`gated` 只是展示用**。上下文由时间线整体重建，所以「主脑那轮看没看到」不再需要靠合并回 contexts 来补——`gated=1` 现在只用于记忆页面与排障，**不要在上下文组装路径上读它**。但**永远不要改写已入账的 `content`**：巡检窗口的判据、时间线剔除当前消息都靠它精确匹配。

**图片的描述不在 `content` 里**。账本里图片永远是 `[图片]` 占位，描述存在 `image_caption` 列，`format_message_line()` 负责拼成 `[图片]（图片内容：…）`。这样既不改 content，又让判定/语境/页面都看得见图。

**巡检是扣押，不是轮询**。`patrol.hold()` 返回一个 Future（`PROCESS` / `KILL`）或 `None`（冷却中，调用方应立即拦下）。同一会话的判定必须串行（`_judging` 集合），窗口起点用 `_last_judged_at` 保证冷却期间的对话不丢。

**插件页面必须显式引入 bridge**。`pages/*/index.html` 的 `<head>` 里要写 `<script src="/api/plugin/page/bridge-sdk.js"></script>`：AstrBot 会给它补访问令牌，且因为已存在就不再往文档末尾注入。若依赖宿主注入，它会晚于页面的内联脚本执行，页面就只能走直接 fetch —— 而插件页 iframe 是 sandbox 的 opaque origin，直连必然 401。

**`_LIVE_SQL` 漏加 = 「她已经忘了」静默失效**。`archived_at = 0 AND condensed_at = 0` 是「这条还在短期记忆里活着」的**唯一**表达。少一个条件，已经清空或已被压进摘要的行就会重新出现在上下文、巡检、压缩与页面里，而且不报任何错。所以账本查询统一走 `ShortTermLedger` 的封装方法（或显式拼 `_LIVE_SQL`），**禁止业务层直写 SQL**；新增查询时先问一句「这条她现在还该记得吗」。

**没收口完就不许清空短期层**。清空会把消息移出常规路径，没压过的内容一旦被清就永远进不了长期记忆。因此 `_run_curation(full_reset=...)` 里：兜底整理、`collapse.failed`、未配置提炼模型、`daily_reset_enabled=False` 四种情况一律跳过清空并告警。

**摘要不豁免清空**。「整理完短期层为空」是这套设计的验收点：当天摘要已先提升为长期记忆，其余摘要与原始消息一起收口。不要给摘要开例外，否则「刚起床时短期记忆是空的」不成立。

**名次的唯一事实源是查询填进去的 `rank` / `rank_total`**。等级由 `curation_level()` 推导、不入库——想加一档就改推导函数，不要新增「等级」列（两个字段早晚会不同步）。凡是要用等级的地方（页面、降级、遗忘）都必须走 `store.board()` / `ranked()` / `list_page()`，它们才会把名次填上。

**生命周期**：耗时 `await`（尤其等模型）之后要检查 `self._is_terminating`；插件卸载后数据库连接会关，`core/common/db.py` 的 `_NullCursor` 负责让迟到的读写静默失效，不要依赖它做正确性保证。

## 测试与验证

**跑任何测试前先探查部署环境**：`docker ps` 确认存在 `astrbot` 容器。存在时必须用容器验证，宿主机环境与运行环境不符。

```bash
# 编译与配置校验
docker exec astrbot sh -lc "cd /AstrBot/data/plugins/astrbot_plugin_shittim && python -m compileall -q ."

# 冒烟测试：把当前代码复制到容器 /tmp 再跑，不要污染运行中的插件目录
docker cp . astrbot:/tmp/astrbot_plugin_shittim
docker exec astrbot sh -lc 'cd /AstrBot && PYTHONPATH=/tmp:/AstrBot python /tmp/xxx_smoke.py'
```

- 需要读真实库时**先复制一份副本**（如 `/tmp/memtest/`）再打开，别直接对运行中的库做写入测试。
- `main.py` 能否导入也要验一遍（`python -c "import astrbot_plugin_shittim.main"`），编译通过不等于导入通过。
- 改完 `_conf_schema.json` 与 i18n 后，用 `python -c "import json; json.load(open(...))"` 校验一次。

## 提交与版本

- 版本号只在 `metadata.yaml` 的 `version` 维护，不要写进 `@register`。
- 影响运行行为的改动提交前递增版本（至少末位），并按语义化规则处理。
- 提交信息用中文摘要，格式 `type(scope): 摘要`，常用 `feat` / `fix` / `refactor` / `perf` / `docs` / `chore`。

## AstrBot 开发文档

AstrBot 的开发文档与参考资料放在 `docs/` 目录下，涉及事件注入、提示词注入、主动消息、请求体组建等主题时先查阅那里。
