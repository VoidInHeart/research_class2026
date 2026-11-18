# 开放集意图识别驱动的多专家执行路由

输入是截至当前用户轮的对话。第一层输出已知意图相关度向量，或判定为 OOS。已知路径的第二层只看向量，返回一个执行专家 ID。OOS 直接走共享基座。选中的执行器使用原始对话和相同的公共工具目录，生成包含 `service`、`method`、`arguments` 的结构化调用。主指标是三项联合完全匹配率。本项目离线生成和评估调用，不连接真实业务 API。

仓库包含可以直接在 CPU 上跑的 SGD 试点、可替换的共享指令基座与三个 LoRA 适配器，以及两种 Jev 在线对照。首次两层实验的数据、结果和局限见 [试点报告](reports/PILOT.md)。

## 离线窗口与未知意图留出实验

主分支的第一层默认只读最近 5 条发言（当前用户发言和之前至多 4 条）。训练、校准、批量及单条预测使用同一窗口，专家执行时仍使用原始完整对话。需要复现旧版时，训练传入 `--recognition-window original`；此前保存的模型重载后继续使用其原窗口。窗口研究支持的是已知意图识别改善，最终调用质量仍按任务分别报告。

在原有本地检索专家上，比较原窗口、最近 5 条发言，以及“长窗口做 OOS 门控，短窗口生成意图向量”。四组未知意图留出、两种编码器合计 24 个组合。报告分别给出已知识别、未知拒识与完整调用成功率，并可从提交的逐条预测离线复算。[实验报告与复现命令](reports/window_holdouts/REPORT.md)

后续实验从 `main` 切研究分支，保留固定基线的配对结果和可复算的评估记录。得到验证的改进连同适用范围合回 `main`，主分支默认配置反映已采纳的方案。

## 本地复现

以下 PowerShell 命令在本仓库根目录运行。此工作区实测解释器是 `D:\Anaconda\envs\research2026\python.exe`。在已激活的该环境内，`python` 即可运行。新环境先安装 `requirements.txt`，MiniLM 还需要 PyTorch 和 Transformers。

```powershell
python scripts/fetch_sgd.py
python -m intentexec run --out artifacts/my-tfidf
python -m intentexec predict --run-dir artifacts/my-tfidf --text 'USER: Find a restaurant in San Jose with American food.'
python -m unittest discover -s tests -v
```

有本地 MiniLM 权重时可替换第一层。下例使用此工作区现有的离线模型缓存。

```powershell
python -m intentexec run --encoder minilm --model-path '../code/models/models--sentence-transformers--all-MiniLM-L6-v2' --out artifacts/my-minilm
python -m intentexec.diagnostics --run-dir artifacts/my-minilm --out reports/my-minilm-diagnostics.json
python -m intentexec.profile --run-dir artifacts/my-minilm --policy matrix --limit 24 --concurrency 1 4 --repeats 2 --out artifacts/my-minilm-profile.json
```

默认试点下载固定 Git 版本的10个 SGD 官方训练分片及相同版本的完整训练 schema，并核对哈希。要扩展到127个训练分片，可运行 `python scripts/fetch_sgd.py --all --out data/sgd-full`，后续命令传 `--sgd-root data/sgd-full`。首次以已知适配条件跑通时可用 `--limit-router 120 --limit-test 100 --live-n 8`，它按公共 ID 哈希抽取请求。每次 `run` 使用新的输出目录，默认执行全量路由监督及测试。

## 数据和监督

- [意图目录](configs/intent_catalog.json) 将同义方法归入同一业务意图，并指定开发未知和最终未知意图。方法名仍作为执行金标的一部分。修改此文件才能研究另一组未知意图；修改随机种子只会改变对话分组分配以及训练随机性。
- 只提取相邻用户、系统轮中可唯一对齐的调用。输入仅拼接至当前用户轮的公开话语，不含标注帧、金标服务、参考调用、下一系统回复或执行返回。训练 schema 的全部26个服务对所有执行器公开，不按当前真值缩小候选。开发和测试属于官方 train 中自建的类别留出划分。
- 专家和识别器只在 fit 上拟合。已训练专家在独立 router 集逐请求执行，以完整调用的实际正确性监督第二层。cal 用于校准第一层绝对相似度 OOS 门槛和已知意图分数温度。划分隔离完整对话及经数字、标点、空白规范化后相同的输入模板；任意释义改写不保证能检出。
- 冻结路由模型后，先从测试公开文本生成并落盘全部选择，再取各执行器的输出和金标评分。正式部署入口 `predict` 每条请求只执行一次。批量计算全部模型的答案用于离线诊断，单独统计其调用量和时间。

## 方法和执行器

| 方法 | 选择行为 |
| --- | --- |
| always_base | 始终调用基座 |
| fixed | OOS 走基座，已知路径用路由监督集整体最佳专家 |
| lookup | 按首选意图查专家历史成功率 |
| matrix | 按完整意图权重对意图×专家成功率矩阵加权。稀疏单元向专家总体均值收缩 |
| mlp | 用完整向量预测各专家成功概率，保留多个专家都成功或都失败的请求 |
| embedding | 共享第一层文本嵌入，直接预测每个专家成功率，检验意图分数的信息损失 |
| matrix_no_gate | 关闭 OOS 旁路，所有请求都按向量选专家 |

`retrieval` 后端用 fit 中的调用模板、公共工具说明和参数抽取生成输出。它提供通用检索兜底与三个侧重不同的执行器（search、transaction、context），用于低成本验证。其指标只反映这个执行池。最终质量比较应在同一强指令基座上训练三个适配器，再以该基座关闭适配器的状态作为兜底。

## LoRA 训练与评估

准备本地指令模型目录（含聊天模板及模型权重），安装与硬件匹配的 PyTorch 和 [LoRA 依赖](requirements-lora.txt)。下面的路径需替换成实际基座位置，示例 8192 为总上下文，要求模型支持该长度。训练与评估使用相同的 schema、划分、种子和提示预算。

```powershell
$baseModel = 'D:/models/instruct-base'
python -m intentexec.lora --base-path $baseModel --output artifacts/adapters-s1118 --device cuda --per-expert 256 --epochs 1 --seed 1118 --max-length 8192 --max-new-tokens 192
python -m intentexec run --encoder minilm --model-path '../code/models/models--sentence-transformers--all-MiniLM-L6-v2' --backend lora --base-path $baseModel --adapters-dir artifacts/adapters-s1118 --device cuda --max-length 8192 --max-new-tokens 192 --out artifacts/lora-s1118
```

每个适配器采用相同条数和训练预算，各自偏向查询、交易或多轮上下文，保留部分共享样本。只监督参考调用 token。推理共享同一份基座，按路由 ID 激活一个适配器或禁用适配器；加载时核对训练集与工具目录指纹。完整目录优先于旧对话历史，裁剪策略保留最后用户轮。只生成文本到 JSON，不发起订票、支付等真实操作。

## Jev 的两种对照

[TypeSafe 官方 Choice 路由实现](https://github.com/TypeSafeAI/typesafe-router/blob/main/REFERENCE.md) 是协议依据。以下命令会对冻结的公共 SGD 测试输入调用托管接口。先在本机环境设置 `TYPESAFE_API_KEY` 和实际可用的 `JEV_MODEL`，使用可追踪版本。费用参数按实验当天产品计费自行提供，缺失项留空。

```powershell
python -m intentexec compare-jev --run-dir artifacts/my-minilm --out artifacts/jev-comparison --model $env:JEV_MODEL --limit 20
```

1. `jev_direct` 直接读对话，从三个专家和基座中选执行者。
2. `jev_vector` 共享识别器；OOS 时跳过 Jev，其他请求将固定顺序的已知意图名与分数传给 Jev，在三个专家中选一个。专家选项附路由开发集能力表。
3. 同一批请求交错运行两种 Jev 和三个本地策略，记录实际选择、执行结果、失败、用量以及决策和端到端耗时。20条最多产生40次 Jev 请求。失败保留并计为未完成，默认不重试。以同一批请求计算对话聚类配对区间。提供实例时价及已确认的 token/逐次费率时，输出每千请求和每次成功的估算费用。收费记录缺失或不确定时返回空值。

## 产物与口径

每个实验目录含可重载模型、分组 ID、路由监督、公开输入、路由选择、离线执行输出、标签、延迟和 `summary.json`。只加载自己生成的 joblib 文件。仓库跟踪代码、配置与汇总报告。原始对话和模型保存在被 Git 忽略的 `data/`、`artifacts/`。

质量报告区分真实已知、真实未知、误回退、误进专家池，另给逐请求事后最佳的上界和按对话重采样区间。工具名称和参数名按 schema 精确比较，参数字符串做 Unicode、大小写和空白规范化，未配置的别名或语义等价改写仍可能计错。`profile` 支持测量本机加载、首请求和不同并发下的排队/服务时间。质量比较使用固定输入和固定执行器；用户等待时间来自真实执行，单位为毫秒。Jev 往返含网络，与本地计算耗时分列。

来源：[SGD 官方数据](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)，[Select-then-Route 2025](https://aclanthology.org/2025.emnlp-industry.28/)，[TRouter 2026](https://aclanthology.org/2026.acl-long.1007/)。本地能力矩阵是意图概率加权的任务适配基线。
