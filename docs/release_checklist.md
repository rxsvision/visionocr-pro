# 发布验收清单：eval_acceptance.py 具名步骤

> 适用范围：每个版本发布（打 `v*` tag 触发 Release Gate）与每次客户交付前的质量验收。
> 核心原则：**CI 只能验证代码卫生与最小依赖测试；检测能力的交付边界必须由
> `scripts/eval_acceptance.py` 在带数据集与权重的完整环境中验收**。本文档把该脚本
> 写成具名步骤：执行点、判决规则、不通过的回退动作。

## 前提条件

验收依赖仓库外资产（均被 gitignore，见 `DEPLOY.md` 第 4 节模型分布架构）：

- GPU 推理环境（CUDA 可用，`torch.cuda.is_available() == True`）
- 完整依赖 `requirements.txt`（CI 的 `requirements-test.txt` 不含 torch/anomalib）
- 验收数据集（按模式选择，路径经 argv 传入，脚本不硬编码任何数据路径）
- 产品 OK 记忆库与 YOLO 门控权重（如涉及产品线交付）

## 执行点（何时跑）

| 场景 | 必跑步骤 | 说明 |
|---|---|---|
| 版本发布：打 tag 前 | 步骤 1 + 步骤 2 基线对照 | Release Gate（`.github/workflows/release.yml`）自动跑 pytest 与卫生门禁，但不含检测能力验收；**打 tag 前必须在本机补跑本清单** |
| 客户交付/换线 | 步骤 3（该产品的模式） | 交付边界 = 该产品的验收指标达到判决规则 |
| 新产品接入投产 | 走 `docs/new_product_onboarding.md` 五步法第 4/5 步 | 本清单 `kolektor` 模式作为其验收脚本示例 |

## 判决规则（预注册，不得事后放宽）

### 通用底线（所有模式）

- 任一指标劣于上一版本同数据集基线且超出下述容差 → 判不通过；
- 判决规则须在跑分**之前**写死（预注册），历史规则记录于 `CHANGELOG.md` 各版本条目；
- 历史参照：KolektorSDD 验收 PASS 记录为 FPR-hold 11.43%→2.86%、Recall 80.77% 不降。

### 各场景判决表

| 场景 | 指标 | 放行条件 |
|---|---|---|
| 引擎/阈值升级（如 dvab 类改动） | Recall / FPR-hold / AUROC | Recall 降幅 ≤ 2 个百分点，且 FPR-hold 或 AUROC 至少一项更优，才允许升级 |
| Prompt/检测策略调优 | Recall / FPR-hold / 平均框数 | Recall ≥ 20% ∧ FPR-hold ≤ 10% ∧ 平均框数 ≤ 3 |
| SubspaceAD 快速换线 | 1-shot Recall@eps=0.10 | ≥ PatchCore 全库基线的 85% |
| 新产品接入（五步法第 5 步） | 已知 NG 集 Recall ∧ OK 验证集误报率 | Recall = 100%（漏检零容忍）∧ 误报率 ≤ ε，双条件同时满足 |
| 成对打光/无标注自举（paired/bootstrap） | 离群排名 + 目检核验 | 无量化放行线：输出供人工目检核验，核验通过方可作为建库/验收输入 |

指标口径（与 `scripts/eval_acceptance.py` 一致）：

- AUROC：图像级 rank-based（含并列处理）
- FPR/Recall：各引擎 NP 阈值（或 legacy fallback）下的图像级判定
- Union OR：任一引擎判 NG 即 NG（零漏检架构）

## 具名步骤

### 步骤 1：发布前基线对照（打 tag 前必跑）

```bash
# mask 标注表面缺陷数据集 (Part*.jpg + Part*_label.bmp), 80/20 划分 (seed 2026)
python scripts/eval_acceptance.py kolektor <数据集根目录> --out eval_kolektor_<版本>.json
```

将 `--out` 结果与上一版本同数据集 JSON 对比：Recall 降幅 ≤ 2pts 且 FPR-hold/AUROC 不退化。

### 步骤 2：换线能力回归（涉及 SubspaceAD 的发布）

```bash
# 1/2/4-shot 建库 vs PatchCore 全库基线
python scripts/eval_acceptance.py subspacead <数据集根目录> --out eval_subspace_<版本>.json
```

### 步骤 3：按交付场景选择模式

| 模式 | 命令 | 数据要求 |
|---|---|---|
| `kolektor` | `python scripts/eval_acceptance.py kolektor <root> [--out X.json]` | mask 标注对 |
| `pcb` | `python scripts/eval_acceptance.py pcb <root> [--out X.json]` | `root/images/<类别>/*.jpg` 为缺陷，`root/PCB_USED/*` 为 OK 建库样本 |
| `paired` | `python scripts/eval_acceptance.py paired <ok_dir> <def_dir> [--name N] [--out X.json]` | 成对打光：ok_dir=对照图，def_dir=缺陷图（按众数尺寸过滤杂项） |
| `bootstrap` | `python scripts/eval_acceptance.py bootstrap <dir> [--name N] [--bank-frac 0.75] [--out X.json]` | 无标注目录，DINOv2 特征质心自举选"近正常"子集建库 |

## 不通过的回退动作

按不通过发生时机分两级：

### A. 发布前不通过 → 不打 tag

1. **停止发布**：不得打 `v*` tag（打了 tag 即触发 Release Gate 归档交付产物）；
2. 定位退化来源（引擎参数/prompt/阈值/依赖升级），修复后重跑步骤 1~3；
3. 修复与验收结果写入 `CHANGELOG.md` 对应版本条目（含预注册规则与实际指标），再走发布。

### B. 交付后发现不达标 → 回退上一版本

1. 执行 Release Gate `rollback-guide` job 打印的回退路径：
   `git fetch --tags` → `git checkout <上一个已发布 tag>`；
2. 按 `DEPLOY.md` 重新部署该修订；
3. 如需数据回滚，使用 `backup_db` 还原（备份位于 `data/backups/`）；
4. 回退版本同样须过步骤 1 基线对照，确认回退目标本身达标。

## 检查清单（可打印）

- [ ] 完整环境就绪（GPU + requirements.txt + 数据集 + 产品记忆库）
- [ ] 判决规则已预注册（写入本次验收记录，不事后放宽）
- [ ] 步骤 1 基线对照：Recall 降幅 ≤ 2pts ∧ FPR-hold/AUROC 不退化
- [ ] 步骤 2/3 按场景跑完，`--out` JSON 已存档
- [ ] 不通过时已执行对应回退动作（不打 tag / 回退上一 tag）
- [ ] 验收指标已记入 CHANGELOG 对应版本条目

## 关联文档与代码

- 验收脚本：`scripts/eval_acceptance.py`（五种模式，路径全经 argv）
- 校准验收示例：`scripts/eval_np_calibration.py`
- 新产品接入（含五步法双条件放行）：`docs/new_product_onboarding.md`
- 发布门禁与回退路径：`.github/workflows/release.yml`（Release Gate / rollback-guide）
- 部署与模型分布：`DEPLOY.md`
- 历史判决规则与验收记录：`CHANGELOG.md`
