# 什亭之枢 AstrBot 插件开发指引

## 插件定位

什亭之枢（Shittim）让 AstrBot 智能体拥有统一人格、跨会话记忆与主动行动能力：
所有私聊与群聊共享同一份记忆，她能自主决定**对谁说、在哪个会话说、什么时候说**。

一句话区分职责：**决断层决定「要不要说」，主脑决定「怎么说」，插件决定「说在哪里」。**

不是「全局作息状态机」——清醒/睡眠、消息队列、睡前告别都还没实现（见 README 路线图），不要照着 README 的愿景去写代码。

## 架构：四条链路

### 拦放链路（接管官方 LLM）

`main.py`：`record_incoming_message`（priority 100，只记账）→ `gate_official_llm`（priority 90）。

判定分两层，边界必须保持清晰：

- `core/decision.py: rule_decide()` —— 同步、零 token。上游指令 / 私聊 / 被点名 / 无文本在这里当场定夺，返回 `None` 表示「交给巡检」。
- `core/patrol.py: PatrolManager` —— 群聊里的普通消息**扣押进窗口**，攒一小段再整体判一次。
- `core/decision.py: judge_window()` —— 面向「一段对话」的模型判定，调用方是巡检器，不直接被事件处理器调用。

放行 = `event.is_at_or_wake_command = True`；拦下 = `event.stop_event()` + `_mark_gated()`。

### 焦点链路（她在哪）

`core/focus.py`，`session_focus` 表。焦点**只约束主动发言**，不约束被动接话。
收消息 / `say` 过去 → 刷新焦点；安静超时或 `leave_session` / `rest` → 失效。

### 记忆链路

`short_term.py`（账本）→ `extractor.py`（提炼，累计 `extract_batch_size` 条触发）→ `memory_store.py`（召回/合并/晋升）→ `maintenance.py`（衰减/遗忘/晋升/合并）。图片消息额外走 `core/vision.py` 转述。

### 行动链路

`main.py` 的 LLM 工具：`list_sessions` / `peek_session` / `say` / `leave_session` / `rest` / `remember` / `recall`。

## 核心模块速查

| 文件 | 职责 | 改动时的注意点 |
| --- | --- | --- |
| `main.py` | 事件处理器、LLM 工具、生命周期 | 优先级常量在文件顶部，改动前先读注释 |
| `core/db.py` | 建表与增量迁移 | 改表结构必须同步 `_DDL`、`_COLUMN_MIGRATIONS`、`SCHEMA_VERSION` |
| `core/short_term.py` | 账本读写与行格式化 | `format_message_line` 是判定/语境块的输入格式 |
| `core/memory_store.py` | 记忆召回与写入 | 召回权重在文件顶部常量 |
| `core/decision.py` | 决断层 | `_WINDOW_PROMPT` 是判定质量的直接来源 |
| `core/patrol.py` | 巡检窗口 | 扣押/替换/冷却/噪声四件事都在这里 |
| `core/injector.py` | 注入块拼装 | 只负责「长什么样」，「放哪里」在 main |
| `core/history.py` | 官方历史回填与合并 | `merge_missing_into_contexts` 依赖 content 精确匹配 |
| `core/vision.py` | 图片转文字 | 引用必须当场取出，临时文件会失效 |
| `core/config.py` | 配置数据类 | 新增配置项须同步 `_conf_schema.json` |
| `web_api.py` | 记忆页面只读接口 | 路由注册在 `main.__init__` 里 |
| `pages/memory/` | 记忆页面 | 单文件 HTML，标题靠 i18n 覆盖 |

## 场景导航

- **调整群聊参与时机** → `core/decision.py: rule_decide()` 与 `core/patrol.py`。窗口/封顶/冷却/噪声都在 `DecisionConfig` 的 `patrol_*`。
- **调整判定质量** → `core/decision.py` 的 `_WINDOW_PROMPT`。记住判据是**一段对话**，不是单条消息。
- **调整注入顺序或内容** → `main.py: inject_memory_context()`。顺序 = 模型读取顺序：长期记忆 → 别处的经历 → 原生历史 → 当前消息（`req.prompt` 永远最后）。
- **改账本结构** → `core/db.py` 三处一起改；已有库靠 `_COLUMN_MIGRATIONS` 补列。
- **新增 LLM 工具** → `main.py` 里 `@filter.llm_tool(name=...)`，docstring 就是给模型看的说明书，要写清「什么时候该用」。
- **新增页面** → `pages/<名字>/index.html` + `web_api.py` 注册路由 + `.astrbot-plugin/i18n/zh-CN.json` 里配 `views.<名字>.title`。
- **调整图片转述** → `core/vision.py`（取图、转 data URL、缓存、模型能力检查）。
- **调整主动发言风控** → `proactive` 配置组 + `main.py` 的 `_check_say_gate()` / `_check_rate_limit()`。

## 关键机制与坑

**事件优先级（数值越大越早）**：记账 100 / 决断 90 / 工具清理 60 / 上下文注入 50 / 结果装饰 -100 / 发送后收口 100。
决断层用 90 是「霸道」取值：被拦下的消息，其它插件（含装在一起的天使之心）也看不到。若与其他插件冲突，调低它（只要仍高于默认 LLM 阶段）。

**记账必须快**。记账是事件链第一环，任何耗时操作（模型调用、下载）都要用 `self._spawn()` 丢后台，否则会把后面所有插件的处理一起拖慢。

**`gated` 与 `merge_missing_into_contexts`**：被拦下的消息主脑看不到，靠 `gated=1` 标记后合并回 `req.contexts`。合并是**按 `role + content` 精确匹配**的，所以**永远不要改写已入账的 `content`** —— 改了就会匹配不上，同一句话会被插入两次。

**图片的描述不在 `content` 里**。账本里图片永远是 `[图片]` 占位，描述存在 `image_caption` 列，`format_message_line()` 负责拼成 `[图片]（图片内容：…）`。这样既不改 content，又让判定/语境/页面都看得见图。

**巡检是扣押，不是轮询**。`patrol.hold()` 返回一个 Future（`PROCESS` / `KILL`）或 `None`（冷却中，调用方应立即拦下）。同一会话的判定必须串行（`_judging` 集合），窗口起点用 `_last_judged_at` 保证冷却期间的对话不丢。

**插件页面必须显式引入 bridge**。`pages/*/index.html` 的 `<head>` 里要写 `<script src="/api/plugin/page/bridge-sdk.js"></script>`：AstrBot 会给它补访问令牌，且因为已存在就不再往文档末尾注入。若依赖宿主注入，它会晚于页面的内联脚本执行，页面就只能走直接 fetch —— 而插件页 iframe 是 sandbox 的 opaque origin，直连必然 401。

**生命周期**：耗时 `await`（尤其等模型）之后要检查 `self._is_terminating`；插件卸载后数据库连接会关，`core/db.py` 的 `_NullCursor` 负责让迟到的读写静默失效，不要依赖它做正确性保证。

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
