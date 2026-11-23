# 执行器升级后，意图向量能否把请求分给更合适的专家

2026-11-23。完整正确要求服务、方法和全部参数同时正确。本轮的旧、新路由都使用上一轮修复后的本地执行器，并采用同一个已校准的 OOS 门。因此两种路由的差值来自放行后选了谁。所有推理都在本地完成。

**有效改进是执行器变更后，用原监督请求重新估计三专家能力表。** 相同的公交购票从 12/218 到 115/218，在独立补充样本上从 2/11 到 9/11。固定采用交易专家也达到这两个结果。保留主分支的三专家结构，提供旧运行的能力表刷新入口。四路矩阵和保守策略作为研究对照归档。

## 比较了什么

- **旧矩阵**沿用修复执行器之前测得的专家成功率。**重估三专家矩阵**用相同的已知路由监督请求重新执行搜索、交易、上下文三个专家，按完整调用的成功率更新能力表。第二层仍将各意图权重乘以这张表，选期望成功率最高的专家。OOS 请求直接走兜底。
- **固定专家**也在旧开发数据上选，各折都是交易专家。它保留第一层的 OOS 旁路，放行后一律选交易专家。**始终兜底**把所有请求交给通用执行器。在开发数据上把兜底也列入固定执行器候选时，五折的最佳固定执行器都是兜底。
- **四路矩阵**允许第二层在已放行的请求上主动选择兜底。另一种**保守策略**只有在已知训练意图内的配对结果支持专家优于兜底，且优势超过小样本惩罚时才选专家。二者都只读取第一层的意图向量，OOS 仍直接走兜底。

原十个 SGD 分片里，共对 598 条不同的旧路由监督请求各运行四个执行器。每一折只取该折允许的已知监督子集，例如公交购票折取 447 条，排除涉及购票的训练对话。将对话和重复模板分组后做五折交叉验证，候选按完整正确数选定，并列时优先兜底。公交、住宿、活动和音乐均选中始终兜底。天气/账户折选四路矩阵，开发成绩为 186/598，始终兜底为 185/598。开发参数、阈值来源和执行池在验证前冻结于 `fd040ee`。[开发锁和逐请求监督](development.json) · [研究协议与合并判断](PROTOCOL.md)

## 原十九片能解释上次的公交错误

| 留出的未知交易 | 数量 | 旧矩阵 | 重估三专家 | 固定交易专家（保留门控） | 始终兜底 | 四路矩阵 |
|---|---:|---:|---:|---:|---:|---:|
| 公交购票 | 218 | 12 | **115** | **115** | 111 | 111 |
| 住宿预订 | 137 | 1 | 1 | 1 | 1 | 1 |
| 活动购票 | 206 | 11 | 11 | 11 | 11 | 11 |

公交门控始终拒绝 4 条、放行 214 条。旧矩阵把放行的购票全部给搜索专家，重估后 212 条给交易专家、2 条给上下文专家。相比旧矩阵，104 条从错变对，1 条从对变错，净增 103 条（+47.25 个百分点，对话配对 95% 区间 [38.32, 55.91]）。相比始终兜底，同一策略救回 13 条，也丢掉 9 条，净增 4 条（+1.83 个百分点，区间 [−2.24, 6.14]）。在这三组未知交易上，固定交易专家与重估矩阵得到的完整正确数相同。

原十九片的公交、住宿、活动整折各含 3,236 条请求。旧矩阵分别答对 328、587、698 条，重估后为 757、748、757 条。对真正的公交查询，旧新均为 0/84，四路事后最佳也为 0/84。真正的住宿查询从 37/353 到 36/353，活动查询均为 11/145。整体收益不能直接解释为查询能力提升。[完整计数、成对得失与区间](results.json)

## 排除两个容易混淆的解释

**专家有互补性。** 在原十九片的公交购票上，如果保留 OOS 决策、只对放行请求事后挑能答对的执行器，最高可答对 130/218。直接走兜底答对 111 条。即便事后根据每条输入首选的已知意图挑一个固定执行器，也只能答对 119 条。这些是用了评估答案的诊断上界。实验中的重估矩阵是 115 条；开发时只看已知意图，无法提前知道新交易上哪条应给哪个模型。[分组事后上界](opportunity.json)

**分流变化与拒识分别计数。** 第一层用原来的最近五条发言和自适应门控，测试 OOS 的召回率与已知误拒均相同。四路候选在放行后选择兜底，只发生在第二层。每个执行器仍沿用上一轮的 SGD 日期假设，这些成绩评价的是相同执行条件下的路由效果。

## 两批新对话的确认结果

预先固定的十片中有 4,132 条可评分请求。做意图范围和历史对话/模板隔离后，五折覆盖 1,473 个不同请求。公交未知只有 1 条，住宿未知为 0。为确认这两类，另用按历史交易密度确定的十三个分片，隔离先前三十九片，覆盖 2,368 个不同请求。两套数据均来自同一版本 SGD 官方训练集的自建留出。补充集刻意靠近交易较密集的分片，以下结果不跨数据集或跨折汇总。

| 确认集与留出场景 | 整折请求数 | 旧矩阵完整正确 | 重估三专家 | 固定交易专家 | 始终兜底 | 开发集选定策略 |
|---|---:|---:|---:|---:|---:|---:|
| 预定十片 · 公交 | 1,038 | 54 | 136 | 134 | 137 | 137 |
| 预定十片 · 住宿 | 1,038 | 107 | 136 | 134 | 137 | 137 |
| 预定十片 · 活动 | 1,038 | 135 | 138 | 136 | 137 | 137 |
| 补充十三片 · 公交 | 2,368 | 148 | 250 | **271** | 210 | 210 |
| 补充十三片 · 住宿 | 2,368 | 202 | 256 | **267** | 210 | 210 |
| 补充十三片 · 活动 | 2,368 | 226 | 249 | **254** | 210 | 210 |

在补充集五种留出设置中，重估矩阵相对旧矩阵整折分别多答对 102、54、23、20、24 条，对话配对 95% 区间均在零以上。但五折都低于固定专家。预定十片中，重估矩阵相对始终兜底的整折差值只有 −1、−1、+1、−1、−1 条。此前原域折在开发集领先兜底一条的四路矩阵，在预定十片与兜底打平，在补充十三片比兜底净多一条，也未表现出稳定优势。[预定十片分组统计](independent/results.json) · [补充十三片分组统计](supplement/results.json)

**专看未知交易。** 预定十片的公交购票从 0/1 到 1/1，活动从 18/63 到 17/63。补充十三片公交为 2→9/11，固定交易专家与始终兜底也各答对 9 条。住宿为 21→21/237，活动为 29→29/138。补充集真正的公交查询仍为 0/191，住宿查询由 44/368 到 47/368，活动查询均为 10/203。改善集中在能力发生显著变化的执行路径；向量路由的额外价值仍需超过固定专家才能成立。

## 使用和复算

普通从头运行 `intentexec run --backend retrieval-grounded` 本来就会重新监督第二层。新增入口用于迁移已保存的旧运行。它继承源识别器、窗口、阈值、专家顺序及原 `router_used` 样本，写到新的目录。输出可用于 `matrix`、`matrix_no_gate` 或 `always_base` 单请求推理。旧 `artifacts/verified-minilm` 使用全历史窗口及原型门，迁移会继承这两个设置。本研究表格另按上轮最近五条、自适应门控的五折模型回放。

```powershell
python -m intentexec.refresh_saved_matrix --run-dir artifacts/verified-minilm --out artifacts/my-refreshed-matrix
python -m intentexec predict --run-dir artifacts/my-refreshed-matrix --policy matrix --text "USER: I need a bus to Sacramento."
```

核验以下三个归档只需项目的 Python 数值计算依赖，无需网络、SGD 原文或模型权重。

```powershell
python -m intentexec.refresh_evaluation replay --out reports/refreshed_routing
python -m intentexec.refresh_holdout replay --out reports/refreshed_routing/independent
python -m intentexec.refresh_holdout replay --out reports/refreshed_routing/supplement
```

完整重跑沿用旧 MiniLM 权重、旧十片监督、冻结门控模型目录和十九片归档。以下每个输出目录均选一个新路径。数据下载脚本访问固定版本的 SGD 公共静态文件。

```powershell
python -m intentexec.refresh_development --out artifacts/my-routing-development.json
python -m intentexec.refresh_evaluation ledger --out artifacts/my-four-executor-labels
python -m intentexec.refresh_evaluation evaluate --development artifacts/my-routing-development.json --ledger artifacts/my-four-executor-labels --out artifacts/my-routing-replay
python -m scripts.fetch_refresh_holdout
python -m intentexec.refresh_holdout prepare --out artifacts/my-routing-independent
python -m intentexec.refresh_holdout evaluate --out artifacts/my-routing-independent --lock artifacts/my-routing-development.json
python -m scripts.fetch_refresh_holdout --supplement
python -m intentexec.refresh_holdout prepare --supplement --out artifacts/my-routing-supplement
python -m intentexec.refresh_holdout evaluate --out artifacts/my-routing-supplement --lock artifacts/my-routing-development.json
```
