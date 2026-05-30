# VLA Dataset Quality Audit Strategy

## 1. Goal

This document defines a conservative automated strategy for auditing LeRobot-style RGB-D VLA training datasets.

The main goal is not to remove every imperfect episode. The main goal is:

1. Automatically drop only objectively invalid data.
2. Mark uncertain semantic problems as review.
3. Preserve all source data and generate a cleaned dataset only from an explicit keep list.

This avoids deleting useful demonstrations by mistake.

## 2. Dataset Assumptions

The expected dataset layout is:

```text
dataset_root/
  data/
    chunk-xxx/file-xxx.parquet
  depth_sidecar/
    observation.images.front_depth/
      episode-xxxxxx/*.png
  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/chunk-000/file-000.parquet
  videos/
    observation.images.front/
      chunk-xxx/file-xxx.mp4
    observation.images.wrist/
      chunk-xxx/file-xxx.mp4
```

The audit should support any dataset with the same LeRobot-style structure, not only the current `determ_lerobot_rgb_depth_vla` dataset.

## 3. Decision Labels

Each episode should receive one of three labels.

```text
keep
  The episode passed all hard checks and has no strong warning.

review
  The episode is suspicious, but the problem is semantic or threshold-based.
  It should not be deleted automatically.

drop
  The episode has objective structural, synchronization, numerical, decoding,
  or sensor defects. It is unsafe for training.
```

The most important rule:

```text
Automatic drop is allowed only for objective failures.
Task failure is review by default, not drop.
```

## 4. Required Output Files

An audit run should create:

```text
quality_report.csv
quality_summary.json
keep_episodes.txt
review_episodes.txt
drop_episodes.txt
```

By default these files are written together under:

```text
quality_audit_reports/<dataset_name>/
```

If `--out-dir` is provided, the audit writes directly to that directory.

Recommended `quality_report.csv` columns:

```text
episode_index
status
length
data_rows
depth_files
drop_reasons
review_reasons
not_checked
metrics_json
```

`reasons` should use stable machine-readable tags, for example:

```text
missing_depth_file
video_decode_failed
timestamp_not_monotonic
action_nan
depth_invalid_high
front_video_frozen
low_motion_review
task_failure_likely_review
```

## 5. Hard Drop Checks

These checks can be used for high-confidence automatic deletion from the training set.

### 5.1 Structure And File Integrity

Drop an episode if any of the following is true:

```text
data parquet for the episode is missing
data parquet cannot be read
episode metadata row is missing
episode length differs from parquet row count
referenced RGB video file is missing
referenced RGB video cannot be opened
depth directory is missing
depth file count differs from episode length
one or more referenced depth PNG files cannot be read
dataset contains junction, symlink, or Windows reparse directories
```

Dataset-level hard failures:

```text
meta/info.json missing or unreadable
meta/tasks.parquet missing or unreadable
meta/episodes/chunk-000/file-000.parquet missing or unreadable
total_episodes differs from episode metadata row count
total_frames differs from sum of episode lengths
total_tasks differs from tasks.parquet row count
```

### 5.2 Index And Synchronization Integrity

Drop an episode if any of the following is true:

```text
episode_index in its data parquet does not match the metadata episode_index
frame_index is not contiguous inside the episode
global index is duplicated or missing
timestamp contains NaN or Inf
timestamp is not monotonic inside the episode
dataset_from_index and dataset_to_index do not match the expected global frame range
```

Use a conservative timestamp rule:

```text
timestamp must be non-decreasing
large timestamp gaps should be review unless they break alignment
negative timestamp jumps should be drop
```

### 5.3 Numerical Validity

Drop an episode if any of the following is true:

```text
action contains NaN or Inf
observation.state contains NaN or Inf
action dimension differs from meta/info.json
observation.state dimension differs from meta/info.json
all action values are constant and all sampled RGB frames are also frozen
```

Action jumps should be handled conservatively:

```text
clear impossible jumps -> drop
large but physically plausible jumps -> review
```

The script should calculate robust per-dimension statistics first, then flag jumps using robust thresholds such as median absolute deviation or high quantiles.

### 5.4 Sensor Failure

Drop an episode if any of the following is true:

```text
front video cannot be decoded
wrist video cannot be decoded
video duration/frame availability is inconsistent with metadata
all sampled RGB frames are black, white, or near-constant
long consecutive sampled RGB frames are exactly or nearly identical
all depth frames are zero, constant, unreadable, or wrong-sized
depth invalid ratio is extremely high for nearly the whole episode
```

Recommended conservative initial thresholds:

```text
black frame:
  RGB mean < 5 and RGB std < 3

white frame:
  RGB mean > 250 and RGB std < 3

near-constant RGB frame:
  RGB std < 2

episode-level RGB sensor failure:
  more than 95% sampled frames satisfy black/white/near-constant rules

frozen video:
  more than 90% sampled adjacent frame pairs have extremely small difference

depth invalid frame:
  invalid ratio > 95%

episode-level depth failure:
  more than 90% sampled depth frames are invalid
```

These thresholds should be treated as conservative defaults and logged in the report.

### 5.5 Exact Duplicate Episodes

Drop one duplicate if two or more episodes are exactly duplicated.

A high-confidence duplicate requires both:

```text
action/state sequence hash is identical
sampled RGB/depth frame hashes are identical
```

Near-duplicates should be review, not drop.

## 6. Review-Only Checks

These checks are useful, but they should not automatically delete data.

### 6.1 Task Failure

Task failure should be review by default.

Examples:

```text
cup not detected in front camera
robot hand does not approach cup
gripper does not interact with target
final frame does not show successful pickup
object slips or is dropped
trajectory appears unsuccessful
```

Reason:

```text
A real attempt that fails can still contain useful state-action information.
Automatic semantic classification is not reliable enough for hard deletion.
```

The audit should output fields such as:

```text
task_success_likely
task_failure_likely
cup_visible_ratio
robot_target_proximity_score
contact_likely
```

But these fields should only affect `review`, unless the episode is also structurally invalid.

### 6.2 Camera Angle And Occlusion

Mark as review if:

```text
camera view is unusual compared with the dataset distribution
cup is outside the field of view
robot arm is mostly outside the field of view
severe human occlusion is detected
image blur is high for a large part of the episode
```

These are not hard drop conditions unless they are extreme and make the sensor effectively unusable.

### 6.3 Low Motion

Mark as review if:

```text
robot state changes are very small
action magnitude is very small
RGB motion is low
gripper does not change
```

Drop only when low action/state motion and frozen video occur together.

## 7. Recommended Audit Pipeline

The automated pipeline should run in this order.

```text
1. Dataset-level structure check
2. Episode metadata consistency check
3. Data parquet read and index check
4. Action/state numerical check
5. Depth file count and depth image check
6. RGB video open/decode check
7. Sensor quality sampling
8. Duplicate detection
9. Review-only semantic checks
10. Report generation
```

The pipeline should stop early only for dataset-level fatal errors. Episode-level errors should be collected into the report so all problems are visible in one run.

## 8. Sampling Policy

To keep the audit efficient, video and depth checks can use sampling.

Recommended default:

```text
sample first frame
sample last frame
sample 32 evenly spaced frames per episode
sample more frames when a possible failure is detected
```

For hard failures like unreadable files, no sampling shortcut should hide the error:

```text
if metadata says a depth PNG should exist, its path must exist
if video metadata references an mp4, the mp4 must be openable
```

## 9. Clean Dataset Generation

The audit script should not delete the source dataset.

Recommended workflow:

```text
1. Run quality audit.
2. Inspect quality_report.csv and review_episodes.txt.
3. Manually edit drop_episodes.txt if any review episodes should also be removed.
4. Build a cleaned dataset from source minus drop_episodes.txt.
5. Write a cleaning manifest into the output dataset.
```

The cleaning manifest should include:

```text
source_dataset
audit_script_version
audit_time
thresholds
kept_episodes
reviewed_episodes
dropped_episodes
drop_reasons
```

## 10. Suggested CLI

Audit:

```powershell
conda activate vla_data_check
cd /d F:\Lerobot\hf_depth_vla_datasets
python .\lerobot_data_quality_audit.py `
  .\determ_lerobot_rgb_depth_vla `
  --out-dir .\quality_audit_reports\determ_lerobot_rgb_depth_vla `
  --depth-check header `
  --video-check decode `
  --video-content-check sample `
  --sample-frames 8 `
  --depth-sample-frames 8
```

Verify existing dataset integrity:

```powershell
conda activate vla_data_check
cd /d F:\Lerobot\hf_depth_vla_datasets
python .\utils\check_lerobot_completeness.py .\determ_lerobot_rgb_depth_vla
```

Generate a cleaned dataset after review:

```powershell
python .\build_clean_lerobot_dataset.py `
  .\determ_lerobot_rgb_depth_vla `
  --drop-list .\quality_audit_reports\determ_lerobot_rgb_depth_vla\drop_episodes.txt `
  --out-dir .\determ_lerobot_rgb_depth_vla_clean `
  --video-mode trim-reencode
```

## 11. Default Policy Summary

Use this policy unless there is a project-specific reason to change it.

```text
DROP:
  missing/unreadable files
  broken metadata
  broken indices
  broken timestamps
  NaN/Inf in action/state
  unreadable video/depth
  whole-episode black/white/frozen RGB
  whole-episode invalid depth
  exact duplicate episode
  junction/symlink/reparse directory in dataset

REVIEW:
  suspected task failure
  cup not visible
  no likely contact
  unusual camera angle
  low but nonzero motion
  blur or occlusion
  near-duplicate episode
  large but not impossible action jump

KEEP:
  all hard checks pass
  no strong review signals
```

The guiding principle is:

```text
False drops are more harmful than false reviews.
Only delete what is objectively invalid.
```

---

# VLA 数据集质量自动审计策略（中文版）

## 1. 目标

本文档定义一套保守的自动化策略，用于审计 LeRobot 风格的 RGB-D VLA 训练数据集。

核心目标不是删除所有“不完美”的 episode，而是：

1. 只自动删除客观无效的数据。
2. 对语义上不确定的问题标记为 review。
3. 保留所有源数据，只根据明确的 keep 列表生成清洗后的数据集。

这样可以避免误删仍然有价值的示教数据。

## 2. 数据集假设

预期的数据集目录结构如下：

```text
dataset_root/
  data/
    chunk-xxx/file-xxx.parquet
  depth_sidecar/
    observation.images.front_depth/
      episode-xxxxxx/*.png
  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/chunk-000/file-000.parquet
  videos/
    observation.images.front/
      chunk-xxx/file-xxx.mp4
    observation.images.wrist/
      chunk-xxx/file-xxx.mp4
```

审计脚本应支持任何具有相同 LeRobot 风格结构的数据集，而不仅仅是当前的 `determ_lerobot_rgb_depth_vla` 数据集。

## 3. 判定标签

每个 episode 应被分配为以下三种标签之一：

```text
keep
  episode 通过所有硬性检查，并且没有强烈异常信号。

review
  episode 可疑，但问题属于语义判断或阈值判断。
  不应自动删除，需要人工快速复核。

drop
  episode 存在客观的结构、同步、数值、解码或传感器缺陷。
  该数据不适合进入训练集。
```

最重要的规则：

```text
只有客观失败才允许自动 drop。
任务失败默认标记为 review，而不是 drop。
```

## 4. 必需输出文件

一次审计运行应生成：

```text
quality_report.csv
quality_summary.json
keep_episodes.txt
review_episodes.txt
drop_episodes.txt
```

默认情况下，这 5 个文件会统一写入：

```text
quality_audit_reports/<dataset_name>/
```

如果命令中显式传入 `--out-dir`，则直接写入该目录。

推荐的 `quality_report.csv` 字段：

```text
episode_index
status
length
data_rows
depth_files
drop_reasons
review_reasons
not_checked
metrics_json
```

`reasons` 应使用稳定、机器可读的标签，例如：

```text
missing_depth_file
video_decode_failed
timestamp_not_monotonic
action_nan
depth_invalid_high
front_video_frozen
low_motion_review
task_failure_likely_review
```

## 5. 硬性 Drop 检查

以下检查可以用于高置信度自动剔除训练数据。

### 5.1 结构与文件完整性

如果出现以下任一情况，应将 episode 标记为 drop：

```text
episode 对应的 data parquet 缺失
data parquet 无法读取
episode metadata 行缺失
episode length 与 parquet 行数不一致
metadata 引用的 RGB video 文件缺失
metadata 引用的 RGB video 无法打开
depth 目录缺失
depth 文件数与 episode length 不一致
一个或多个 depth PNG 无法读取
数据集中存在 junction、symlink 或 Windows reparse directory
```

数据集级别的硬性失败：

```text
meta/info.json 缺失或无法读取
meta/tasks.parquet 缺失或无法读取
meta/episodes/chunk-000/file-000.parquet 缺失或无法读取
total_episodes 与 episode metadata 行数不一致
total_frames 与所有 episode length 之和不一致
total_tasks 与 tasks.parquet 行数不一致
```

### 5.2 索引与同步完整性

如果出现以下任一情况，应将 episode 标记为 drop：

```text
data parquet 中的 episode_index 与 metadata 中的 episode_index 不一致
episode 内部 frame_index 不连续
全局 index 存在重复或缺失
timestamp 包含 NaN 或 Inf
episode 内 timestamp 非单调递增
dataset_from_index 和 dataset_to_index 与预期全局帧范围不一致
```

timestamp 使用保守规则：

```text
timestamp 必须非递减
较大的 timestamp gap 如果不破坏对齐，应标记 review
timestamp 出现负向跳变，应标记 drop
```

### 5.3 数值有效性

如果出现以下任一情况，应将 episode 标记为 drop：

```text
action 包含 NaN 或 Inf
observation.state 包含 NaN 或 Inf
action 维度与 meta/info.json 不一致
observation.state 维度与 meta/info.json 不一致
action 全程恒定，并且采样 RGB 帧也冻结
```

action 突变应保守处理：

```text
明确不可能的突变 -> drop
较大但物理上仍可能的突变 -> review
```

脚本应先计算每个 action/state 维度的稳健统计量，再使用 median absolute deviation 或高分位数等稳健阈值标记异常突变。

### 5.4 传感器故障

如果出现以下任一情况，应将 episode 标记为 drop：

```text
front video 无法解码
wrist video 无法解码
video 时长或可读取帧数与 metadata 明显不一致
所有采样 RGB 帧为黑屏、白屏或近似常数
长时间连续采样 RGB 帧完全相同或几乎相同
所有 depth 帧为 0、常数、不可读或尺寸错误
整个 episode 中 depth invalid ratio 极高
```

推荐的保守初始阈值：

```text
黑屏帧:
  RGB mean < 5 且 RGB std < 3

白屏帧:
  RGB mean > 250 且 RGB std < 3

近似常数 RGB 帧:
  RGB std < 2

episode 级 RGB 传感器失败:
  超过 95% 的采样帧满足黑屏、白屏或近似常数规则

冻结视频:
  超过 90% 的相邻采样帧差异极小

depth 无效帧:
  invalid ratio > 95%

episode 级 depth 失败:
  超过 90% 的采样 depth 帧无效
```

这些阈值应被视为保守默认值，并写入审计报告。

### 5.5 完全重复的 Episode

如果两个或多个 episode 是完全重复的，可以自动 drop 其中的重复副本。

高置信度重复需要同时满足：

```text
action/state 序列 hash 完全一致
采样 RGB/depth 帧 hash 完全一致
```

近似重复只应标记为 review，不应自动 drop。

## 6. 仅 Review 的检查

以下检查有用，但不应自动删除数据。

### 6.1 任务失败

任务失败默认应标记为 review。

示例：

```text
front camera 中未检测到杯子
机器人手臂没有靠近杯子
夹爪没有与目标物体交互
最后一帧看不出成功抓起
物体滑落或掉落
轨迹看起来没有完成任务
```

原因：

```text
真实尝试但失败的 episode 仍可能包含有用的状态-动作信息。
自动语义分类的可靠性不足以作为硬删除依据。
```

审计脚本可以输出以下字段：

```text
task_success_likely
task_failure_likely
cup_visible_ratio
robot_target_proximity_score
contact_likely
```

但这些字段只应影响 `review`，除非该 episode 同时存在结构性无效问题。

### 6.2 相机角度与遮挡

如果出现以下情况，应标记为 review：

```text
相机视角相对数据集整体分布明显异常
杯子在视野外
机器人手臂大部分在视野外
检测到严重人为遮挡
episode 中大部分图像严重模糊
```

这些情况不是硬性 drop 条件，除非严重到让传感器数据基本不可用。

### 6.3 低运动量

如果出现以下情况，应标记为 review：

```text
机器人 state 变化非常小
action 幅度非常小
RGB 运动量很低
gripper 基本没有变化
```

只有当低 action/state 运动量与视频冻结同时出现时，才应 drop。

## 7. 推荐审计流程

自动化流程建议按以下顺序执行：

```text
1. 数据集级结构检查
2. episode metadata 一致性检查
3. data parquet 读取与索引检查
4. action/state 数值检查
5. depth 文件数与 depth 图像检查
6. RGB video 打开与解码检查
7. 传感器质量采样检查
8. 重复 episode 检测
9. 仅 review 的语义检查
10. 报告生成
```

只有数据集级 fatal error 才应提前停止。episode 级错误应尽量全部收集到报告中，方便一次性查看所有问题。

## 8. 采样策略

为了提高审计效率，video 和 depth 检查可以使用采样。

推荐默认策略：

```text
采样第一帧
采样最后一帧
每个 episode 均匀采样 32 帧
当检测到潜在故障时增加采样帧数
```

对于不可读文件这类硬错误，采样不应掩盖问题：

```text
如果 metadata 表示某个 depth PNG 应存在，则该路径必须存在
如果 video metadata 引用了某个 mp4，则该 mp4 必须可打开
```

## 9. 生成清洗后的数据集

审计脚本不应删除源数据集。

推荐流程：

```text
1. 运行质量审计。
2. 查看 quality_report.csv 和 review_episodes.txt。
3. 如果人工确认某些 review episodes 也应删除，就把它们加入 drop_episodes.txt。
4. 用 source dataset 减去 drop_episodes.txt 的方式生成清洗后的数据集。
5. 在输出数据集中写入 cleaning manifest。
```

cleaning manifest 应包含：

```text
source_dataset
audit_script_version
audit_time
thresholds
kept_episodes
reviewed_episodes
dropped_episodes
drop_reasons
```

## 10. 建议命令行接口

审计：

```powershell
conda activate vla_data_check
cd /d F:\Lerobot\hf_depth_vla_datasets
python .\lerobot_data_quality_audit.py `
  .\determ_lerobot_rgb_depth_vla `
  --out-dir .\quality_audit_reports\determ_lerobot_rgb_depth_vla `
  --depth-check header `
  --video-check decode `
  --video-content-check sample `
  --sample-frames 8 `
  --depth-sample-frames 8
```

验证现有数据集完整性：

```powershell
conda activate vla_data_check
cd /d F:\Lerobot\hf_depth_vla_datasets
python .\utils\check_lerobot_completeness.py .\determ_lerobot_rgb_depth_vla
```

人工复核后生成清洗数据集：

```powershell
python .\build_clean_lerobot_dataset.py `
  .\determ_lerobot_rgb_depth_vla `
  --drop-list .\quality_audit_reports\determ_lerobot_rgb_depth_vla\drop_episodes.txt `
  --out-dir .\determ_lerobot_rgb_depth_vla_clean `
  --video-mode trim-reencode
```

## 11. 默认策略总结

除非项目有特殊要求，否则使用以下默认策略。

```text
DROP:
  文件缺失或不可读
  metadata 损坏
  索引损坏
  timestamp 损坏
  action/state 中存在 NaN 或 Inf
  video/depth 不可读
  整个 episode RGB 黑屏、白屏或冻结
  整个 episode depth 无效
  完全重复的 episode
  数据集中存在 junction、symlink 或 reparse directory

REVIEW:
  疑似任务失败
  杯子不可见
  疑似没有接触
  相机角度异常
  低但非零运动量
  模糊或遮挡
  近似重复 episode
  较大但并非不可能的 action 突变

KEEP:
  所有硬性检查通过
  没有强烈 review 信号
```

指导原则：

```text
误删好数据比多标记 review 更有害。
只删除客观无效的数据。
```
