# 新产品接入指南：OK 库登记与零漏检验收

> 适用范围：每个新接入 VisionOCR Pro 的产品（型号/料号）。
> 核心原则：**零漏检由"异常检测引擎 + 产品级 OK 记忆库"保证**，而不是由零样本检测器保证。
> Grounding DINO 零样本检测仅用于宏观结构冷启动（无 OK 库时的过渡手段）——真实产线
> 实测表明它对表面缺陷（划痕/凹坑/油污等）在默认阈值下召回为 0，降低阈值只会导致
> 框洪水与类别噪声，没有可用工作点。

## 为什么必须先有 OK 样本

PatchCore 与 DINOv2 异常检测是"单类分类"：它们学习的是"这个产品正常时长什么样"，
任何显著偏离正常分布的区域即判 NG。没有 OK 样本就没有记忆库，引擎无法工作，
此时系统只能退回零样本检测（不可承担零漏检责任）。

**结论：NG 样本再多也建不了库。** 仅有 NG 图的数据集只能用于检出敏感性回归，
不能作为验收依据。

## 登记五步法

### 第 1 步：采集 OK 样本（门槛 ≥50，推荐 ≥100）

拍摄要求：

- 与产线检测工位完全一致的光源、角度、焦距、背景；
- 覆盖正常波动：不同批次、轻微位置偏移、正常加工纹理方向变化；
- 剔除混入的不良品（采集时人工过一遍）；
- 分辨率与实际检测一致，不要缩放后再入库。

样本量与误报控制的关系：NP 校准阈值的最小粒度约为 1/n（n 为校准样本数）。
样本少于 100 张时系统会自动输出警告（`core/np_calibration.py`），提示单库实际
误报率可能偏离目标值。**50 张是能建库的下限，100+ 张才能得到稳定的误报率控制。**

### 第 2 步：建库（PatchCore + DINOv2 双库）

UI 路径：QC Tab → 工程师模式 → 「少样本注册 (PatchCore)」→ 选择/新建产品 →
多文件上传 OK 样本 → 「📦 注册建库」。建库时自动留出尾部 20% 作 NP 校准集
（校准图不入 bank）。

工程接口（`core/anomaly_bank.py`）：

```python
from core.anomaly_bank import register_ok_samples

result = register_ok_samples(registry, product_name, ok_image_paths)
# PatchCore 主库必建; DINOv2 副库 best-effort (失败不阻塞)
```

建库产物按产品名隔离存储，`load_product_bank()` 按产品加载。

### 第 3 步：NP 阈值校准（误报率上界 ε）

用 OK 样本的异常分数分布校准判定阈值 τ，保证 P(正常件分数 > τ) ≤ ε：

```python
from core.np_calibration import NPCalibrator

cal = NPCalibrator(epsilon=0.02)   # 目标: 正常件误报率 ≤ 2%
cal.fit(normal_scores)             # n<100 时自动输出粒度警告
```

ε 的工业含义：最多允许 ε 比例的正常件被判 NG（过杀率上限）。
按产线节拍与复检成本选择，常见取值 1%~5%。

#### 校准协议（n_cal 扩充，方案 §6.2）

建库时自动留出的校准集只有上传量的 20%（10~30 张上传 → n_cal 仅 3~6）。
n_cal 过小会导致两个问题：阈值粒度 ~1/n 过粗（实际误报率偏离 ε），以及
分阶段融合停留在 Stage 1（纯 OR，误报偏高）。解决办法是**建库后补采独立
校准图重标定**：

1. 补采 **≥30 张独立 OK 图**——不得是建库用图；建议变换光照/角度拍 3 组，
   覆盖产线真实正常波动；
2. UI：QC Tab → 工程师模式 → 「📐 校准协议」→ 选产品 → 上传校准图
   （可选附 NG 样本做 Recall 回归）→ 执行；
3. 系统对 PatchCore/DINOv2 逐源重标定 NP 阈值（保证不变：P(正常>τ) ≤ ε），
   重标定写回 bank，校准集存档 `data/calibration/{产品}/{时间戳}/`；
4. 验收报告给出 n_cal、τ（旧→新）、融合阶段变化与 NG 回归结果。

融合阶段联动：n_cal ≥10 → Stage 2（双源互证，单源孤证转人工复核），
n_cal ≥50 → Stage 3（附加漂移监控）。工程接口见
`core/calibration_protocol.py`（`recalibrate_product` / `format_report_md`）。

### 第 4 步：NG 回归（零漏检验证）

对该产品**所有已知 NG 样本**跑完整检测链（Union：PatchCore + DINOv2 + 产品门控
YOLO + 零样本冷启动），要求 **Recall = 100%**。任何漏检都必须通过调库/调 ε/补
样本解决，而不是放行。

### 第 5 步：放行门槛（双条件）

| 条件 | 指标 | 要求 |
|---|---|---|
| 已知 NG 集 | Recall | = 100%（漏检零容忍） |
| OK 验证集 | 误报率 | ≤ ε（NP 校准目标） |

两条件同时满足，产品方可投入产线自动判定；否则回到第 1/3 步补样本或调 ε。

## 可选：YOLO 结构缺陷通道（标注式，§5.4）

异常检测（PatchCore/DINOv2）对微观**结构**缺陷（缺孔/短路/毛刺/开路等）
判别力有限；若该产品有 ≥50 张可标注缺陷图，可启用 YOLO 少样本通道作为
Union 第三检测源：

1. 标注：框出结构缺陷（LabelImg 等，VOC XML 或 YOLO txt 均可）；
2. 数据准备：`python finetune/prepare_pcb_data.py --src <数据集路径>`
   （PCB 流程通用化后同样适用其他产品）；
3. 一键精调：`python finetune/train_yolo.py`（默认 **YOLO11n** 基线，
   `--model yolov8n/s/m/x` 兼容旧权重；微缺陷建议 `--imgsz 1280`）；
4. 权重落盘：将 `finetune/output_yolo/*/weights/best.pt` 复制为
   `models/yolo/{产品名}.pt`，产品门控（`core/yolo_products.py`）自动启用。

约束：

- **无通用兜底权重**——未训练的产品不会套用其他产品的 YOLO 权重
  （跨域误报防护），Union 自动跳过该源，不影响其余检测链；
- ultralytics 为 AGPL-3.0，仅在启用本通道时需安装（`requirements.txt`
  中已注释，按需 `pip install "ultralytics>=8.3"`）；
- 投产验收仍走第 4/5 步：YOLO 源并入 Union 后重跑 NG 回归。

## 常见误区（实测验证）

1. **"我有一批 NG 图，先跑起来看看"** —— NG-only 数据建不了 OK 库，跑出来的
   只有零样本结果（表面缺陷召回≈0），会得出"系统不行"的错误结论。正确顺序：
   先采 OK 建库，再用 NG 回归。
2. **用文件名/目录名当标签** —— 弱标签不可靠：实测七个历史项目数据集的
   目录名标签与目视核验存在成组矛盾（如"划伤"目录下无可见划伤）。验收必须
   以人工复核过的样本为准。
3. **降 GDINO 阈值追召回** —— 阈值 0.3→0.15 虽恢复召回，但每图涌出 10~42 个框、
   七类缺陷名几乎同时命中，无分数间隙可分，只会制造误报。表面缺陷交给异常引擎。
4. **VLM 判了 OK 就是 OK** —— VLM（qwen3-vl 等）是解释与复核辅助，实测对明显
   裂纹/密集划痕召回可信，但对弱纹理缺陷会漏；不承担零漏检责任。

## 检查清单（可打印）

- [ ] OK 样本 ≥50 张（推荐 ≥100），与产线同条件拍摄
- [ ] 建库成功（PatchCore 主库 + DINOv2 副库状态确认）
- [ ] NP 校准完成，ε 取值已按产线复检成本确定
- [ ] 校准协议已执行：独立校准图 ≥30 张，n_cal≥30（融合 Stage ≥2）
- [ ] 已知 NG 集 Recall = 100%
- [ ] OK 验证集误报率 ≤ ε
- [ ] 产品名称/料号与 MES 工单字段一致
- [ ] （可选）YOLO 结构缺陷通道：≥50 张标注图精调完成，
      权重已落盘 `models/yolo/{产品名}.pt` 并复跑 NG 回归

## 关联文档与代码

- 部署与环境：`DEPLOY.md`、`scripts/doctor.py`
- 建库：`core/anomaly_bank.py`（`register_ok_samples` / `load_product_bank`）
- 校准：`core/np_calibration.py`（`NPCalibrator`，n<100 自动警告）
- 校准协议：`core/calibration_protocol.py`（`recalibrate_product`，§6.2 n_cal 扩充）
- 分阶段融合：`core/fusion.py`（Stage 1/2/3 与漂移监控）
- YOLO 结构缺陷通道：`finetune/train_yolo.py`（YOLO11 基线训练）、
  `core/yolo_products.py`（按产品权重门控）
- UI 入口：`ui/tab_qc.py`（产品登记 / 校准协议）
- 验收脚本示例：`scripts/eval_np_calibration.py`、`scripts/eval_acceptance.py`
  （`fusion55` / `calibration` 模式）
- 发布验收清单（执行点/判决规则/回退动作）：`docs/release_checklist.md`
