# FAST 论文复现项目说明

本项目用于复现论文：`FAST: Topology-Aware Frequency-Domain Distribution Matching for Coreset Selection`。

当前项目已经从“最小骨架”发展成一个可运行的 FAST 复现平台，已经具备：
- CIFAR-10 数据读取与 PCA 特征提取
- 多尺度图构建、MST 增强、谱分解
- 连续 coreset 参数化与首版 FAST 联合优化
- selected indices 导出
- 下游 ResNet18 / ResNet50 训练与测试
- AFL / PDAS / PD-CFD / graph-aware assignment 的多轮升级版本
- repeat 汇总、表格生成、ablation 入口

但也要明确：
- 当前项目更适合“渐进式复现、模块验证、pilot 实验、最小论文对齐实验”
- 还不能直接认为已经完整复现论文全部数值
- full CIFAR-10 / 更大规模设置下，图构建和 assignment 仍然可能成为瓶颈

---

## 1. 项目目标

这个项目的核心目标是实现 FAST 的两阶段流程：

1. 在完整训练集上做 coreset selection，得到一个离散子集
2. 只用这个子集训练下游分类器，并在测试集上评估精度

也就是说，FAST 本身不是分类器，而是一个“数据子集选择方法”。

---

## 2. 当前推荐的最小目标实验

当前最适合先对齐的实验是：
- Dataset: CIFAR-10
- Keep Ratio: 10%
- Backbone: ResNet18
- Method: FAST

原因：
- 当前代码对 CIFAR-10 支持最完整
- 图构建、谱分解、FAST 优化和下游训练都已接通
- 配置、repeat、结果保存和表格生成都已经有对应入口
- 这是最适合先做“最小论文对齐”的设置

---

## 3. 项目目录说明

### 3.1 根目录

- `README.md`
  - 当前说明文档。
- `requirements.txt`
  - 项目最小依赖列表。
- `data/`
  - 数据目录。当前主要存放 CIFAR-10。
- `configs/`
  - 实验配置文件。
- `src/`
  - 项目核心源码。
- `scripts/`
  - 命令行实验入口、汇总脚本、表格脚本、ablation 入口。
- `tests/`
  - 单元测试、smoke test 和小型联调脚本。
- `outputs/`
  - 运行结果、日志、selected indices、repeat 汇总、表格、ablation 输出。

---

### 3.2 configs/

当前配置文件：

- `configs/cifar10_fast.yaml`
  - 当前默认主配置。
  - 目前等价于 formal 配置，用于接近论文 Table 1 的正式设置。
- `configs/cifar10_fast_debug.yaml`
  - 小规模 debug 配置。
  - 用于快速验证整条 pipeline 能不能跑通。
- `configs/cifar10_fast_pilot.yaml`
  - 小规模 pilot 配置。
  - 用于比 debug 更认真一点的实验，但仍然避免直接上 full 50k。
- `configs/cifar10_fast_formal.yaml`
  - 当前最接近正式实验的配置。
  - 适合用来做 CIFAR-10 / 10% / ResNet18 的最小论文对齐尝试。

配置文件大致包含这些部分：
- `seed`
- `device`
- `num_threads`
- `data`
- `coreset`
- `graph`
- `sampling`
- `assignment`
- `optimize`
- `experiment`
- `eval`

---

### 3.3 src/

#### `src/data/`

- `cifar.py`
  - CIFAR-10 数据读取与预处理模块。
  - 支持：
    - train/test 加载
    - flatten
    - standardize
    - PCA
    - 返回原图像、标签、构图特征 `X`

主要函数：
- `load_cifar10_split()`
- `standardize_train_test_features()`
- `apply_pca_train_test()`
- `prepare_cifar10_data()`

---

#### `src/graph/`

- `knn_graph.py`
  - 多尺度 kNN 图构建。
  - 支持：
    - `rho_i`
    - `sigma_i`
    - directed fuzzy graph
    - fuzzy union
    - 多尺度融合
    - MST 增强
    - 图统计输出

主要函数：
- `build_single_scale_graph()`
- `fuse_multiscale_graphs()`
- `build_mst_graph()`
- `add_mst_edges()`
- `build_multiscale_knn_graph()`

- `spectral.py`
  - 谱分解模块。
  - 支持：
    - `L_sym`
    - `eigsh`
    - 跳过 near-zero eigenvalues
    - smallest non-zero eigenvectors 提取

主要函数：
- `compute_symmetric_normalized_laplacian()`
- `spectral_decomposition()`

- `assign.py`
  - 连续 coreset proxy 到离散样本的 assignment。
  - 支持：
    - full Hungarian baseline
    - pruned candidate mode
    - graph-aware cost

主要函数：
- `compute_cost_matrix()`
- `compute_pruned_cost_matrix()`
- `hungarian_match()`

---

#### `src/losses/`

- `graph_losses.py`
  - `Lmatch` 和 `Lgraph`

主要函数：
- `compute_match_loss()`
- `compute_graph_loss()`

- `dpp.py`
  - diversity loss 的首版实现
  - 基于 RFF 和 `-log det(K)`

主要函数：
- `compute_dpp_loss()`

- `pdcfd.py`
  - PD-CFD 核心模块
  - 已明确区分：
    - ECF
    - amplitude discrepancy
    - phase discrepancy
    - `lambda_phi`
    - total per-frequency loss

主要函数：
- `empirical_characteristic_function()`
- `amplitude_discrepancy()`
- `phase_discrepancy()`
- `pd_cfd_loss()`

---

#### `src/sampling/`

- `anisotropic_freq.py`
  - AFL：Anisotropic Frequency Library
  - 当前版本支持：
    - low / medium / high band
    - band-specific anisotropic scaling
    - 候选 scale 搜索
    - 基于当前 discrepancy 的评分

主要函数：
- `build_anisotropic_frequency_library()`

- `pdas.py`
  - PDAS：Progressive Discrepancy-Aware Sampling
  - 当前版本支持：
    - `tau_t` 控制候选池
    - per-frequency LCF
    - diversity penalty
    - discrepancy-aware greedy selection

主要函数：
- `select_progressive_frequencies()`

---

#### `src/optimize/`

- `optimize_coreset.py`
  - FAST 首版联合优化主循环
  - 串联：
    - 初始化 `Y`
    - assignment
    - `Lmatch`
    - `Lgraph`
    - `Ldiv`
    - PDAS
    - PD-CFD
    - 最终 selected indices 导出

主要函数：
- `initialize_coreset_variable()`
- `optimize_coreset()`
- `export_selected_subset()`
- `save_selected_indices()`

---

#### `src/eval/`

- `train_classifier.py`
  - 下游分类训练与评估入口
  - 当前支持：
    - ResNet18 / ResNet50
    - SGD / Adam
    - scheduler: `none / multistep / cosine`
    - train/test 日志
    - result 保存

主要函数：
- `build_backbone()`
- `build_optimizer()`
- `build_scheduler()`
- `train_one_epoch()`
- `evaluate_classifier()`
- `train_classifier_on_subset()`
- `compare_subset_strategies()`

---

#### `src/utils/`

- `seed.py`
  - 随机种子管理
- `io.py`
  - YAML 读取、目录创建等小工具

---

### 3.4 scripts/

- `run_fast_pipeline.py`
  - 主 pipeline 入口
  - 支持：
    - `--config`
    - `--debug`
    - `--method`
    - `--keep-ratio`
    - `--seed`
    - `--repeat`

- `summarize_repeat_results.py`
  - 汇总 repeat 结果
  - 输出：
    - `summary.json`
    - `summary.md`

- `generate_table1_markdown.py`
  - 将多个实验结果转成 markdown 表格
  - 用于最小 Table 1 风格输出

- `run_metric_ablation.py`
  - distribution metric ablation
  - 当前支持：
    - `MSE`
    - `KL`
    - `CE`
    - `PD-CFD`

- `run_frequency_strategy_ablation.py`
  - frequency strategy ablation
  - 当前支持：
    - progressive discrepancy-aware
    - non-progressive discrepancy-aware
    - progressive uniform
    - non-progressive uniform
    - collinear selection

---

### 3.5 tests/

`tests/` 里包含两类内容：

1. 单元测试
2. 轻量联调脚本 / debug 脚本

主要测试文件：
- `test_cifar_data.py`
- `test_knn_graph.py`
- `test_spectral.py`
- `test_assignment_and_graph_losses.py`
- `test_dpp.py`
- `test_pdcfd.py`
- `test_pdas.py`
- `test_optimize_init.py`
- `test_optimize_joint.py`
- `test_optimize_export.py`
- `test_train_classifier_smoke.py`
- `test_repeat_summary.py`
- `test_smoke.py`

主要联调/调试脚本：
- `run_small_spectral_pipeline.py`
- `run_small_fast_optimization.py`
- `run_loss_forward_check.py`
- `run_debug_afl.py`
- `run_debug_pdcfd.py`

---

### 3.6 outputs/

`outputs/` 存放所有运行结果。

典型内容包括：
- `selected_indices.npy`
- `selected_indices_stats.json`
- `run_summary.json`
- `classifier_result.json`
- `aggregate_*.json`
- `summary.json`
- `summary.md`
- `table.md`
- `ablations/...`

建议不要手动依赖某个固定 debug 输出，而是按 `method / keep_ratio / seed / run_xx` 的目录层级去找结果。

---

## 4. 安装方式

建议使用 Python 3.10 + PyTorch。

安装依赖：

```bash
pip install -r requirements.txt
```

如果你在服务器上运行，建议先显式限制线程，例如：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

当前主脚本 `scripts/run_fast_pipeline.py` 内也已经做了保守线程限制，但服务器上手动设一下通常更稳。

---

## 5. 如何使用

### 5.1 先跑全部测试

```bash
python -m unittest discover -s tests
```

如果你只想先看训练入口是否还正常：

```bash
python -m unittest tests.test_train_classifier_smoke -v
```

---

### 5.2 跑最小 debug pipeline

```bash
python scripts/run_fast_pipeline.py --debug
```

作用：
- 用非常小的 CIFAR-10 子集跑通全链路
- 检查图构建、谱分解、FAST 优化、selected indices 导出、下游训练是否都能走通

适合场景：
- 新机器第一次检查环境
- 改完核心代码后的 smoke test

---

### 5.3 跑 pilot 实验

```bash
python scripts/run_fast_pipeline.py --config configs/cifar10_fast_pilot.yaml --keep-ratio 0.1 --method fast
```

当前 `pilot` 配置不是 full 50k，而是显式限制样本数的小规模 pilot，用于：
- 比 debug 更认真地看一遍结果
- 避免一上来就触发 full 图构建的内存 / 时间问题

---

### 5.4 跑 formal 配置

```bash
python scripts/run_fast_pipeline.py --config configs/cifar10_fast_formal.yaml --keep-ratio 0.1 --method fast --repeat 3
```

注意：
- 这表示“配置目标接近正式实验”
- 不代表当前整套实现已经对 full CIFAR-10 的图构建完全做了规模优化
- 如果卡住或耗时过长，优先怀疑图构建和 assignment 规模问题，而不是训练入口

---

### 5.5 跑 random baseline

```bash
python scripts/run_fast_pipeline.py --config configs/cifar10_fast_pilot.yaml --keep-ratio 0.1 --method random
```

用途：
- 和 FAST 做最小 baseline 对比
- 验证下游训练与结果保存逻辑

---

### 5.6 汇总 repeat 结果

```bash
python scripts/summarize_repeat_results.py --input-dir outputs/full/method_fast/keep_0.100/seed_42
```

输出：
- `summary.json`
- `summary.md`

---

### 5.7 生成最小 Table 1 风格 markdown 表格

```bash
python scripts/generate_table1_markdown.py \
  --results outputs/table_smoke/fast.json outputs/table_smoke/random.json \
  --output-dir outputs/table_smoke
```

输出：
- `table.md`

表头为：

```md
| Method | Dataset | Keep Ratio | Backbone | Mean Acc | Std |
```

---

### 5.8 跑 distribution metric ablation

```bash
python scripts/run_metric_ablation.py --debug
```

当前比较：
- `MSE`
- `KL`
- `CE`
- `PD-CFD`

结果目录：
- `outputs/ablations/distribution_metric/...`

---

### 5.9 跑 frequency strategy ablation

```bash
python scripts/run_frequency_strategy_ablation.py --debug
```

当前比较：
- progressive discrepancy-aware
- non-progressive discrepancy-aware
- progressive uniform
- non-progressive uniform
- collinear selection

结果目录：
- `outputs/ablations/frequency_strategy/...`

---

## 6. 当前推荐工作流

建议按这个顺序用：

1. `python -m unittest discover -s tests`
2. `python scripts/run_fast_pipeline.py --debug`
3. `python scripts/run_fast_pipeline.py --config configs/cifar10_fast_pilot.yaml --keep-ratio 0.1 --method fast`
4. `python scripts/run_fast_pipeline.py --config configs/cifar10_fast_pilot.yaml --keep-ratio 0.1 --method random`
5. 结果稳定后，再考虑 formal 配置和 repeat

---

## 7. 当前项目状态

### 已经比较完整的部分
- CIFAR-10 数据读取和 PCA
- 多尺度图构建主链
- 谱分解
- 连续 coreset 初始化与首版联合优化
- selected indices 导出
- 下游 ResNet18 / ResNet50 训练
- repeat 汇总和表格生成

### 仍然是 MVP / pilot 的部分
- full CIFAR-10 大规模图构建效率
- 大规模 assignment 效率
- AFL / PDAS / PD-CFD 与论文最终实现的完全一致性
- 正式论文级实验系统化管理

### 当前最可能的瓶颈
- 图构建中的全对距离 / MST
- assignment 在更大规模下的代价矩阵和 Hungarian

---

## 8. 常见问题

### Q1. 为什么 `pilot` 不直接跑 50000 全量？
因为当前项目的方法模块已经能跑，但 full CIFAR-10 下图构建和 assignment 仍然是明显的规模瓶颈。`pilot` 的目标是“更认真地跑通”，不是“直接 full 正式复现”。

### Q2. 服务器上为什么会报 OpenBLAS / OMP 相关错误？
因为 sklearn / NumPy / OpenBLAS / PyTorch 在某些服务器环境里会叠加并行，导致线程过多、内存区域分配过多。当前主脚本里已经做了保守线程限制，但服务器上仍建议手动 export。

### Q3. 为什么 debug accuracy 很低？
因为 debug 模式只跑极小样本、极少迭代、极少 batch，它的目标是检查链路，不是看最终精度。

### Q4. 当前是不是已经复现出论文数值了？
还没有。当前更准确的说法是：
- 项目已经形成“可运行的 FAST 复现平台”
- 已经具备对齐最小目标实验的基础设施
- 但距离论文完整数值对齐还有明显的算法细节和规模问题需要继续推进

---

## 9. 结果文件怎么看

最常看的几个文件：

- `run_summary.json`
  - 单次运行的总体摘要
- `classifier_result.json`
  - 下游训练日志和最终精度
- `selected_indices.npy`
  - 当前选择出的子集索引
- `graph_stats.json`
  - 图构建统计信息
- `summary.json`
  - repeat 汇总
- `table.md`
  - 表格输出

建议优先看：
1. `graph_stats.json`
2. `run_summary.json`
3. `classifier_result.json`

---

## 10. 当前最重要的提醒

如果你的目标是“先把当前项目稳定跑起来”，请优先用：
- `debug`
- `pilot`

不要一上来就默认认为 `formal/full` 配置已经等价于“能稳定跑完 full CIFAR-10 正式实验”。

当前项目最真实的定位是：
- 方法模块已经比较齐
- 实验基础设施已经具备
- 但 full 论文级规模还需要进一步优化与对齐