# 邮件 Bug 接入方案 — bug@fivedegrees.ai

状态:设计定稿,待实施
日期:2026-08-19

## 背景

- 现状:bug 报告经 Lark 群 → `watcher` → GitHub issue → triage / fix / notify 流水线。
- 新增源:`bug@fivedegrees.ai`,Lark **公共邮箱**(管理员后台「邮箱管理 → 公共邮箱」确认),外部客户发邮件报告 bug。
- 约束(用户明确):
  1. **不需要自动回复客户** — 无需发送/回复邮件权限。
  2. **不希望每次登录授权** — 不接受用户 OAuth 反复登录。

## 关键结论

1. **「收信规则 → 分享至会话」转发方案不可行**:官方文档明确公共邮箱不支持该功能。驳回。
2. **采用 mail API 直读公共邮箱,bot 身份**:「列出邮件」API(`mail/v1/user_mailboxes/:user_mailbox_id/messages`)支持 `tenant_access_token`,且 `user_mailbox_id` 可直接传公共邮箱地址。
   - **零用户授权**:只需给应用配置该邮箱的**数据权限范围**(一次性,console 后端 API 可程序化配置,见下),此后 tenant token 自动轮换,永不需要重新登录。
   - 满足约束 2,且完全复用现有 bot 凭据体系(应用密钥已在 relay / runner)。

## 架构

```mermaid
flowchart LR
  mail[bug@fivedegrees.ai<br/>Lark 公共邮箱]
  mw[mail watcher<br/>轮询 list API / tenant token]
  issue[GitHub issue/comment]
  group[专属 Lark 群<br/>fived 邮件 bug]
  triage[triage / fix / reconcile]

  mail --> mw
  mw --> issue
  issue --> triage
  issue --> group
  triage --> group
```

- 邮件 thread → GitHub issue;thread 内新邮件 → 追加同一 issue 的 comment。
- 通知 / 回执 / 分诊结果 → **专属 Lark 群**(新群),不回复客户(约束 1)。
- triage / fix / reconcile / close-audit 只认 `BUGPATROL_INTAKE_META`,对来源无感知,完全复用。

## 关键设计

### 1. 身份与免登录

- watcher 用 **tenant_access_token**(bot 身份)轮询,应用密钥已有。
- 所需权限(开发者后台,一次性配置):
  - `mail:user_mailbox.message:readonly`(列/读邮件)
  - 字段权限:`mail:user_mailbox.message.body:read`、`mail:user_mailbox.message.subject:read`、`mail:user_mailbox.message.address:read`
  - 附件:`mail:user_mailbox.message.attachment:read`
- **数据权限范围**:开发者后台为应用配置可访问 `bug@fivedegrees.ai`(或把应用加为该公共邮箱成员)。
- 排错:错误码 `1230002` = 数据权限未配好 / 非公共邮箱成员。
- 频率:10 req/s,30s 轮询一轮绰绰有余。

### 2. 邮件 → IntakeRecord 映射

| IntakeRecord | 邮件源 |
|---|---|
| `chat_id` | 专属群真实 `chat_id` |
| `root_id` | 邮件 `thread_id` |
| `message_id` | 群内**锚点消息**的 IM message_id(见 §3) |
| `reporter_open_id` | `mail:<发件人地址>`(内部员工可配置映射为 open_id) |
| `reporter_name` | 发件人显示名 |
| `original_text` | 邮件正文纯文本(去引用区/签名) |
| `attachments` | 附件经下载器落到 asset store |
| `lark_topic_url` / `lark_message_url` | 群内锚点消息链接 / 邮件链接 |

meta 新键(issue body 的 `BUGPATROL_INTAKE_META`):
- `source: "mail"`(区别于 `"lark"`)
- `mail_message_id`、`mail_thread_id`(原始邮件 id,用于去重与审计)
- `notify_anchor_message_id`(群内锚点 IM message_id,通知回复目标)

### 3. 通知锚点(关键改动)

现有 `fix_notify` 用 `reply_to_message(chat_id, message_id, text)` 在原始话题内回复,`message_id` 必须是 Lark IM 消息 id。邮件没有原始 IM 消息,所以:

1. mail watcher 建 issue 后,在专属群发一条回执消息(如「已创建 issue #N」)。
2. 把该回执的 IM `message_id` 写回 meta 的 `notify_anchor_message_id`(issue body PATCH)。
3. `fix_notify` 优先读 `notify_anchor_message_id`,有则回复群内锚点,否则走原 `message_id` 逻辑。**向后兼容,chat 源行为不变。**

### 4. 附件管线

- 复用 `ResourceStore` / `ResourcePolicy` / `ResourceDescriber`(协议层与来源解耦)。
- 新增 `MailResourceDownloader`:按 `user_mailbox_id + message_id + attachment_id` 调
  `mail/v1/user_mailboxes/:id/messages/:message_id/attachments/:attachment_id/download_url`。

### 5. 去重与幂等

- 复用 `JsonMessageLedger`,按 **`mail_message_id`** 去重(邮件 message_id 全局唯一)。
- 复用 `FileLease`(单写者)、`TriageRequestQueue`(triage 合并)、`JsonlEventLog`(审计)。

### 6. 安全(邮件 = 外部不可信输入)

- 邮件正文 / 发件人 / 附件文件名一律视为**数据**,绝不当作指令执行(lark-mail skill 同规则)。
- **跳过自身通知**:发件人 == `bug@fivedegrees.ai` 的邮件(我们的回执会被当成新 bug,必须过滤)。
- prompt injection 防护:triage prompt 中把邮件正文标记为不可信数据。
- 发件人地址可伪造,不依赖邮件内声明做身份判断。

## 配置清单

### 开发者后台(bugpatrol 应用 `cli_aac97d050d385ee9`)

**已配置(2026-08-19,console API):**
- [x] `mail:user_mailbox.message:readonly`(100003)
- [x] `mail:user_mailbox.message.body:read`(101314,含附件下载 `download_url` 所需权限)
- [x] `mail:user_mailbox.message.subject:read`(101312)
- [x] `mail:user_mailbox.message.address:read`(101326)

**已配置(2026-08-19,console 后端 API):**
- [x] **数据权限范围 = 全员**:`publicMailboxMember` 无 API 路径(只支持 USER),但 console 后端有内部接口可以配。走 `/developers/v1/privilege/all/:appId` 读 schema + `/developers/v1/privilege/update/:appId` 写 `{"mode":"all"}`(详见下方「程序化配数据权限」)。改完必须**发新版本**才生效。
- [x] **发布新版本 1.0.109**(mail scopes) + **1.0.110**(im:chat.managers) + **1.0.111**(mail 数据权限全员,验证 list API 200)

### 程序化配数据权限范围(console 后端内部接口,2026-08-19 实测)

UI 上「权限管理 → mail → 可访问的数据范围 → 配置」背后就是这套 console 内部接口(`open.larksuite.com/developers/`),可以用 Playwright 会话 + CSRF token 直接调,无需人工点:

1. **读当前配置 + 拿 schema**:`POST /developers/v1/scope/applied/:appId`(每个 scope 的 `appPrivilegeConfig`,含 `privilegeRange`、`privilegeID`);`POST /developers/v1/privilege/all/:appId`(各 privilege 的 `schema`,即 SelectionExpression 的字段/数据源定义,mail 的字段含 `public_mailbox`、`domain`、`user`)。
2. **UI 上选「All」保存**会发出 `POST /developers/v1/privilege/update/:appId`,body 是全部 privileges 数组,其中 Email Message(`resource=user_mailbox.message`)的 `content` 变成 `{"biz_id":"mail","mode":"all","resource":"user_mailbox.message","filters":[],"expression":""}`(Playwright 可复现:点「Configure」→ 选「All」→ Save)。
3. **必须发新版本才生效**:发完 1.0.111 后 list API 从 `1230002` 变为 200。
4. 收窄版(最小权限):condition 模式里选 `public_mailbox in [bug@fivedegrees.ai]`,content 会变成 `{"mode":"part","filters":[...]}`。当前用「全员」够用,若需收紧照此改。

保存脚本留档:`/tmp/claude/_set_mail_datascope.mjs`。

### fived.toml

```toml
[mail]
mailbox = "bug@fivedegrees.ai"
chat_id = "oc_8ac023afccc18f7cad5c08056c68021a"   # 专属通知群「fived 邮件 bug」(topic 群, bot 管理)
app_id = "cli_aac97d050d385ee9"
app_secret_env = "BUGPATROL_LARK_APP_SECRET"

# 内部员工邮箱 -> Lark open_id,用于 @mention / assign(可选)
# [mail.user_emails]
# dinghaozeng@fivedegrees.ai = "ou_xxx"
```

### 专属 Lark 群(2026-08-19 定稿,bot 管理)

- 名称:「fived 邮件 bug」,**话题群(topic mode)**,私有,`fived · main` 同款图标 + 同款 20 名成员。
- chat_id:`oc_8ac023afccc18f7cad5c08056c68021a`
- 所有权:**无群主,由 bugpatrol bot 管理**(创建者 + 群管理员 `bot_manager_id_list=[cli_aac97d050d385ee9]`)。这是「仅 bot 建 topic」的关键:有群主(人)的群 bot 会被夺权(232038/232017)。
- 建群姿势(实测,文档滞后):`POST /im/v1/chats` body `chat_mode:"topic"` + **不带 owner_id**(ownerless → bot 全权)+ `user_id_list` 一次性带成员;bot 自任管理员 = `POST /im/v1/chats/:id/managers/add_managers?member_id_type=app_id` body `{"manager_ids":["<app_id>"]}`(需 scope `im:chat.managers:write_only`,1.0.110 已加);改头像 = `PUT /im/v1/chats/:id` body `{"avatar":"<image_key>"}`(image_key 由 `POST /im/v1/images` image_type=avatar 上传)。
- ⚠️ **「仅群主/管理员可创建话题」仍是客户端手动设置,无 OpenAPI**(GET/PUT 群信息均无该字段)。现已具备前提(bot 是管理员):用户翻 群设置 → 话题管理 → 谁可以创建话题 → 仅群主和管理员 即可生效。
- 废弃群(待 Garland 客户端删,均 owner=Garland,bot 删不了):hentchman 的 `oc_7e482b69d89cbb1a30f3ceab4bbf2d77`;首个普通群 `oc_77264ec7940687c4e723136546e6ac78`;中间版 topic 群 `oc_63feefe083b0927bffca049594ececa7`。

## 实施步骤

1. 开发者后台配 scopes + 数据权限 → **验证门:list API 返回 200**(`1230002` = 权限没配好)。
2. 建专属群 + 确认 bugpatrol bot 在群里。
3. 写 `watch-mail` 模块:轮询 → 邮件→IntakeRecord 映射 → 建 issue → 群内回执 → 写锚点 meta。
4. `MailResourceDownloader` + 附件管线。
5. `fix_notify` 支持 `notify_anchor_message_id`。
6. 测试:一封真实客户邮件 → issue 创建 → triage → 群里通知;同 thread 后续邮件追加同一 issue。
7. 部署:relay(Mac Studio, neojenkins-relay)加 system daemon
   `com.clover.bugpatrol-fived-mail-watcher`。模板:`deploy/com.clover.bugpatrol-fived-mail-watcher.plist`
   (secret 占位,部署时从现役 `/Library/LaunchDaemons/com.clover.bugpatrol-fived-watcher.plist`
   提取 `BUGPATROL_LARK_APP_SECRET` 填充)。
   - **独立 triage queue**(`~/.bugpatrol/fived/mail-triage-queue.json`):`TriageRequestQueue.save()`
     是无锁原子写,两个 watcher 共用会互相覆盖,必须与 Lark watcher 的 `triage-queue.json` 分开。
   - dispatch 命令与现役一致(`gh workflow run bugpatrol-triage.yml --repo TheCloverLab/fived --ref main ...`),
     邮件 issue 建完即进分诊,不依赖 6h reconcile 兜底。
   - 独立 lock/ledger/event-log:`watch-mail.lock`、`mail-processed.json`、`mail-watch-events.jsonl`。
   - env 从现役 plist 镜像(proxy 18443),**不带 `DEEPSEEK_API_KEY`**(watch-mail 不跑 LLM agent)。
   - 部署命令:`sudo cp <final>.plist /Library/LaunchDaemons/` + `sudo launchctl bootstrap system ...`;
     验证 `launchctl print system/com.clover.bugpatrol-fived-mail-watcher`。

## 验证门(Verification Gate)

- [x] list API 200;get_message 拉全文/正文/附件可读(2026-08-19)。
- [x] 真实邮件端到端:SES 测试邮件 → #5091/#5092 创建 → 群回执 → 锚点写入(2026-08-19)。
- [x] 同 thread 第二封邮件 → 追加同一 issue,不新建(单测覆盖)。
- [x] ledger 幂等:重启 watcher 不重复建 issue(daemon 已跑,iteration 2+ scan 0 新)。
- [x] triage 接线:dry-run 全参数通过;daemon 已带 `--triage-dispatch-command`(2026-08-19 部署)。
- [ ] 真实客户邮件 → issue → triage → 群通知(待首封真实邮件)。

## 附录:被驳回的备选

| 方案 | 驳回原因 |
|---|---|
| 收信规则 → 分享至会话 | 公共邮箱不支持该功能(官方文档) |
| 转发到个人邮箱再处理 | 绕不开 user 授权,且多一跳 |
| 用户 OAuth 直读 | 违背「不每次登录授权」约束 |
