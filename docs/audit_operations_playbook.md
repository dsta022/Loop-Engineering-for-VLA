# 数据审计操作手册 (Audit Operations Playbook)

本手册定义日常审计的标准操作流程。核心原则:

- **确定性层是每次都跑的数据审计**：快速，merge/clean 之后都要重跑。
- **语义层是结构干净之后才需要进一步进行检查**：只在结构问题清零后补跑，用
  `--semantic-only` 复用上一轮结构报告，不重复付视频解码的成本。
- **定版前跑一次完整流水线**：发布用的报告必须由单次完整审计产出。

所有命令在仓库根目录、`vladata_check` conda 环境中执行:

~~~powershell
conda activate vladata_check
~~~

下文以 `.\lerobot_derek_depth` 为数据集示例，审计输出默认写入
`.\quality_audit_reports\lerobot_derek_depth\`。

---

## 流程总览

~~~mermaid
flowchart TD
    A[采集 / 合并数据集] --> B[阶段 1: 快速结构审计]
    B --> C{有结构性 drop 或全局错误?}
    C -- 有 --> D[修复数据 / 重新合并] --> B
    C -- 无 --> E[阶段 2: 完整确定性审计]
    E --> F{有结构性 drop?}
    F -- 有 --> D
    F -- 无 --> G[阶段 3: --semantic-only 语义补跑]
    G --> H[阶段 4: 人工复查 review 列表]
    H --> I[阶段 5: 语言标注]
    I --> J[阶段 6: 清洗构建 + 定版完整审计]
    J --> K[上传 Hugging Face]
~~~

---

## 阶段 1：快速结构审计（每次数据变动后）

采集或合并完成后立刻跑一遍最便宜的检查，捕获系统性问题（索引错位、文件缺失、
元数据不一致），不做视频解码:

~~~powershell
python .\pipeline\data_quality_audit.py .\lerobot_derek_depth `
  --video-check ffprobe `
  --video-content-check none `
  --depth-check header
~~~

判读:

- `Status: fail` 或有 `GLOBAL_ERROR` / 大量 `DROP` → 说明是系统性问题（常见于
  merge 出错），**先修数据再往下走**，不要进入语义层。
- 通过后进入阶段 2。

## 阶段 2：完整确定性审计（结构基本干净后）

加上完整解码与采样内容检查（黑帧/白帧/冻结帧/深度异常）:

~~~powershell
python .\pipeline\data_quality_audit.py .\lerobot_derek_depth `
  --depth-check header `
  --video-check decode `
  --video-content-check sample `
  --sample-frames 8 `
  --depth-sample-frames 8
~~~

输出在 `.\quality_audit_reports\lerobot_derek_depth\`:

- `quality_report.csv` — 逐 episode 明细（本轮 semantic 列为空，属预期）；
- `keep_episodes.txt` / `review_episodes.txt` / `drop_episodes.txt`；
- `quality_summary.json` — 全局摘要与阈值。

有 drop 就回去修数据并重跑本阶段。**这份报告是阶段 3 的复用基础，跑完后不要
再改动数据集，否则复用会被安全检查拒绝。**

## 阶段 3：语义补跑（只付语义评估的成本）

结构干净后，用 `--semantic-only` 在上一轮报告之上只跑语义评估。结构结论原样
保留，已是结构 drop 的 episode 自动跳过语义推理:

~~~powershell
python .\pipeline\data_quality_audit.py .\lerobot_derek_depth `
  --semantic-only `
  --semantic-evaluator my_quality_backend:create_evaluator `
  --semantic-config .\semantic_evaluator.json `
  --semantic-failure-threshold 0.5 `
  --semantic-pass-threshold 0.9
~~~

需要任务级匹配与阈值覆盖时加 `--task-profile-dir .\task_profiles`（Profile 用
`python .\tools\task_semantic_studio.py` 制作）。

内置后端：不写自定义插件时，可直接用仓库自带的 VLA checkpoint 动作一致性
评估器，把训练好的策略目录（如 `checkpoint for audit\pretrained_model`）当
数据质检器用:

~~~powershell
python .\pipeline\data_quality_audit.py .\lerobot_derek_depth `
  --semantic-only `
  --semantic-evaluator semantic_backends.vla_checkpoint_evaluator:create_evaluator `
  --semantic-config .\configs\vla_checkpoint_evaluator.example.json
~~~

该后端在采样锚点让策略按（图像+关节状态+任务文本）预测动作块，与示教实际
执行的动作对比打分（1.0 = 完全复现示教，0.0 = 不比保持不动好），并自动用
"换错指令"（动词反转/方向/颜色/物体替换）重打分得到反事实边际——换错指令
后预测同样好的 episode 说明其语言标签没有接地，进 review。注意：它需要策略
运行时（`pip install lerobot` 及策略自身依赖），请在训练环境执行本阶段；
低分也可能只是当前策略的分布外数据，阈值须在人工标注集上校准后再信任
自动 pass。

说明:

- 默认复用 `<out-dir>\quality_report.csv`；要指定别的报告用
  `--reuse-report <路径>\quality_report.csv`。
- 合并后的报告**覆盖写回**同一目录：semantic 列填充完成，keep/review/drop
  列表按合并结果重算，`quality_summary.json` 里 `audit_mode: semantic_reuse`
  记录了复用来源。
- 调阈值、换 evaluator 后直接重复本阶段即可，结构检查不会重跑。
- 安全边界：episode 集合、每集长度、上一轮 summary 里的数据集路径都会复核，
  任何不一致直接判 fail——此时说明数据变过，回到阶段 1。
- 本模式禁止 `--feedback-sidecar`（反馈对齐属于结构检查，见下文反馈流程）。

## 阶段 4：人工复查

1. 打开 `review_episodes.txt`，逐条对照 `quality_report.csv` 的
   `review_reasons` / `semantic_failure_stage` / `semantic_score` 看视频确认。
2. 确认确实要丢弃的，把 episode 号补进 `drop_episodes.txt`。
3. 确认可用的 review episode，写入一份单独的批准清单（如
   `approved_episodes.txt`，每行一个 episode 号），供语言标注阶段使用。
4. 语义 pass 的高分 episode 也要**持续抽样人工核验**，防止分数未校准导致的
   假阳性。
5. 把人工判定写成 `episode_index,label` 的 CSV 后，用
   `python .	ools	hreshold_calibration.py --quality-report <报告目录> --labels <标签CSV>`
   得到每条 review 规则的精确率/召回率和推荐的 failure/pass 阈值。

## 阶段 5：语言标注（只花在可用数据上）

~~~powershell
python .\pipeline\trajectory_language_enrichment.py .\lerobot_derek_depth `
  --task-profile-dir .\task_profiles `
  --annotator my_language_backend:create_annotator `
  --annotator-config .\language_annotator.json `
  --approved-episodes .\approved_episodes.txt
~~~

默认只处理 `keep`；`--approved-episodes` 显式纳入人工批准的 review episode，
`drop` 永远拒绝。输出写在 `language_enrichment_outputs\`，不改源数据。也可以
用 `python .\tools\trajectory_language_studio.py` 图形界面配置。

## 阶段 6：清洗构建 + 定版完整审计

构建干净数据集:

~~~powershell
python .\pipeline\dataset_clean_build.py `
  .\lerobot_derek_depth `
  --drop-list .\quality_audit_reports\lerobot_derek_depth\drop_episodes.txt `
  --out-dir .\lerobot_derek_depth_clean `
  --video-mode trim-reencode
python .\tools\dataset_completeness_check.py .\lerobot_derek_depth_clean
~~~

**定版审计：对清洗后的数据集单次跑完整流水线（结构 + 语义一起），产出发布用
的最终报告，不使用 `--semantic-only`:**

~~~powershell
python .\pipeline\data_quality_audit.py .\lerobot_derek_depth_clean `
  --depth-check header `
  --video-check decode `
  --video-content-check sample `
  --sample-frames 8 `
  --depth-sample-frames 8 `
  --semantic-evaluator my_quality_backend:create_evaluator `
  --semantic-config .\semantic_evaluator.json `
  --semantic-failure-threshold 0.5 `
  --semantic-pass-threshold 0.9
~~~

预期 `Status: pass`（或仅剩已人工确认可接受的 review）。然后上传:

~~~powershell
$env:HF_XET_HIGH_PERFORMANCE="1"
python .\pipeline\hf_push.py --dry-run
python .\pipeline\hf_push.py --dataset-dir .\lerobot_derek_depth_clean --dataset-name lerobot_derek_depth_clean
~~~

---

## 反馈数据的特别规则

反馈 sidecar 对齐检查属于**结构层**，必须在阶段 1/2 带上，不能在
`--semantic-only` 模式中使用:

~~~powershell
python .\pipeline\data_quality_audit.py .\data\so100_rgbd_feedback `
  --feedback-sidecar .\feedback_outputs\so100_rgbd_feedback\session-001 `
  --depth-check header `
  --video-check decode `
  --video-content-check sample
~~~

之后语义补跑照常:

~~~powershell
python .\pipeline\data_quality_audit.py .\data\so100_rgbd_feedback `
  --semantic-only `
  --semantic-evaluator my_quality_backend:create_evaluator
~~~

新采反馈 `review_status=pending` 会强制进 review；人工批准写入单独清单后再走
`feedback_dataset_build.py`（见 README 的 Closed-Loop Policy Improvement 节）。

---

## 快速参考

| 场景 | 命令要点 |
| --- | --- |
| 采集/合并后第一遍 | `--video-check ffprobe --video-content-check none` |
| 结构定稿 | `--video-check decode --video-content-check sample` |
| 补跑/重跑语义 | `--semantic-only --semantic-evaluator ...` |
| 换阈值重算语义 | 重复 `--semantic-only`，改两个 threshold 参数 |
| 复用别处的报告 | 加 `--reuse-report <路径>\quality_report.csv` |
| 定版发布 | 单次完整审计（结构+语义），不用 `--semantic-only` |
| 反馈数据 | `--feedback-sidecar` 只在结构阶段用 |

常见失败与处置:

- `prior_report_episode_mismatch` / `prior_report_length_mismatch` /
  `prior_report_dataset_mismatch`：数据集在阶段 2 之后被改过或报告拿错了 →
  回到阶段 1 重跑结构审计。
- `missing_prior_report`：还没跑过结构审计，或 `--out-dir` 和之前不一致。
- 语义列全是 `error` / evaluator 异常：评估器配置问题；evaluator 报错只会进
  review，不会误放行数据。
