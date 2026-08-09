# Phase X1.5 G1/G2 与 DBLP 重试执行报告

## 执行边界

本轮只执行文献检索与引用网络扩展，不运行 GPU、训练或论文正文改写。所有外部返回均作为不受信任的 bibliographic data 处理；G1/G2 产物与主 C1 corpus 隔离，尚未合并进 `retrieval_records.json`。

## DBLP 低频重试

- 重试轮次：round 1，batch 0，3 条查询；每条查询使用协议要求的 3–5 秒确定性抖动。
- 失败页：14 → 11；成功的 3 条查询已写回 retrieval snapshot，并重新应用 `manual_adjudications.json`。
- 当前状态：`retrieval_cutoff.json.status=OPEN`，尚未达到关闭条件。
- 剩余 11 条查询已分为 3 个后续 batch；后续 batch 具有 600 秒等待标记，不在本轮立即执行。
- 替代源当前覆盖：OpenAlex 17 页成功、Crossref 17 页成功；DBLP 仍有 6 页成功、11 页失败。

## G1/G2 引用扩展

### G0 seed 对齐

- 协议 seed：13 个；OpenAlex 成功匹配：12 个。
- 未匹配 seed：`A Comprehensive Analysis of Negative Sampling in Knowledge Graph Embedding`。
- G1 backward：12 个匹配 seed 产生 173 条本地 `referenced_works` 边。

### Forward 任务

聚合快照位于 `output/results/evidence_audit_x1_5/snowball/`：

- G1 forward：12/12 个 parent work 均有返回，997 条记录；4 个 parent 的第一页超过 200 条，状态为 `truncated=true`。
- G2 forward：6/6 个候选 component 均有返回，542 条记录；1 个 parent 的第一页超过 200 条，状态为 `truncated=true`。
- 聚合 forward：18 页、1,539 条记录、0 个请求失败；总体状态仍为 `PARTIAL`，因为有 5 个 parent 尚未完成分页。
- 另外保留了可重放的首批隔离目录：
  - `snowball/g1_forward_batch0/`：1 页、36 条记录。
  - `snowball/g2_forward_batch0/`：1 页、134 条记录。

## 门禁影响

本轮没有释放 C1 novelty gate。当前 `c1_gate_closure.json` 仍为 `UNRESOLVED`，阻塞项包括人工裁决队列、DBLP 未完成/失败页和候选来源核验状态。G1/G2 记录必须先去重、主题筛选和人工裁决，才能进入 novelty evidence matrix；不得直接用于 RETAIN/NARROW/REFRAME/DROP。

## 下一步

1. 按 600 秒等待标记继续 DBLP 剩余 batch，三轮后按协议关闭无法解析页。
2. 对 5 个 truncated parent 继续 OpenAlex cursor 分页，并重新尝试缺失 G0 seed 的 OpenAlex 标题/DOI 对齐。
3. 将 G1/G2 记录作为独立输入做去重、自动主题筛选和人工裁决；完成后再重跑 Part 3–4。
