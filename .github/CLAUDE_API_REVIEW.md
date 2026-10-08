# Claude API 审查

`Claude API Review` 使用 Anthropic API 计费，模型固定为 `claude-opus-5-5`，
effort 为 `max`。它不使用 Claude 订阅的 OAuth token，也不切换 Claude 网站上
托管 Code Review 的计费设置。托管审查是另一个独立服务；如需避免两套服务重复审查，
请在 Claude 组织的 Code review 设置中关闭该仓库的托管自动审查。

仓库管理员需要设置 GitHub Actions secret `CLAUDE_REVIEW_API_KEY`，以及 variable
`ANTHROPIC_WORKSPACE_ID`。密钥仅存放在 GitHub Secrets，不得写入源码、评论或日志。

针对 `1006-stable`、来自本仓库分支的非草稿 PR，在创建、推送、重新打开或转为正式
审查时自动运行。执行者还需通过官方 action 的仓库写权限检查。外部 fork 的 PR 不会
自动使用此密钥。要手动运行，为 PR 添加 `claude-api-review` 标签；再次运行可以先
移除再添加标签，或重跑已有 Actions 任务。

每次审查设置 CLI 预算 `$2`、最多 30 轮及 20 分钟超时。预算由 CLI 在请求之间检查，
不是 API 账户的硬消费上限；最后一个请求可能使实际费用略超预算。工作流的运行摘要
记录 CLI 估算费用及返回的模型，最终费用以 Anthropic Console 的 API 账单为准。

审查结果由 GitHub Actions 机器人写入 PR 评论。它只做静态代码审查，不替代测试、
科学验证或人工合并决策，不具有推送代码的权限。测试应在不携带 API 密钥的独立任务
中运行。
