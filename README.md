# 基于 EM（GMM）的用户行为分群与分析

## 项目背景

在电商与零售场景中，用户消费行为差异巨大：有人高频高客单，有人久未复购。若对所有用户采用同一套运营策略，成本高、效果差。**用户分群** 的目标是把行为相似的客户归为一类，再针对每类制定差异化触达（召回、升单、会员维护等）。

本项目以 **RFM 模型**（Recency 最近消费、Frequency 购买频次、Monetary 消费金额）刻画用户价值与活跃度，采用 **高斯混合模型（GMM）** 完成聚类。GMM 的参数估计依赖 **EM（Expectation-Maximization）算法**，可在存在重叠分布时给出 **软分配概率**（用户属于各群体的置信度），比硬聚类更适合业务解释与风险评估。

系统按 **「训练—展示」分离** 设计：

| 角色 | 职责 |
|------|------|
| **分析人员** | 本地清洗数据、训练 GMM、导出 `output/` 产物 |
| **客户/业务人员** | 通过 Streamlit 输入 RFM，获得单用户/批量判别结果；在「整体报告」查看本批 CSV 的汇总与图表 |

客户端 **不执行训练**，仅加载已导出的模型与统计结果，保证部署简单、界面稳定。

---

## 项目结构

```
基于 EM 算法的用户行为分群与分析/
├── data/                          # 原始与测试数据
│   ├── online_retail.csv          # 训练用交易明细（需自行准备或脚本下载）
│   └── rfm_test_data_100.csv      # 批量判别示例/测试
├── output/                        # 训练产物（app.py 依赖此目录）
│   ├── manifest.json              # 多周期模型清单
│   ├── 1y/、6m/、3m/、1m/         # 四个统计周期各自模型与报告数据
│   └── （每周期目录含 gmm_model.pkl、model_meta.json、*.csv）
├── core/                          # 核心后端逻辑
│   ├── periods.py                 # 统计周期配置（近1年/半年/季度/月）
│   ├── segments.py                # 统一群体编号 1/2/3 与名称对照
│   ├── dataset.py                 # 数据清洗、按窗口截取、RFM、标准化
│   ├── model.py                   # GMM 训练与 BIC/AIC/轮廓系数评估
│   └── predictor.py               # 单用户/批量在线预测与业务解读
├── ui/                            # 前端与可视化
│   ├── layout.py                  # 侧栏周期选择与模型加载
│   ├── batch_report.py            # 批量判别整体报告
│   ├── manual_page.py             # 使用手册页面内容
│   ├── charts.py                  # 图表
│   └── components.py              # 文案组件
├── utils/
│   └── io.py                      # 加载 output 产物
├── scripts/
│   ├── download_data.py           # 从 UCI 下载并转为 CSV
│   └── run_train.py               # 【分析人员】本地训练入口
├── run_dev.bat                    # 一键：训练 → 启动 Streamlit（Windows）
├── app.py                         # 【客户】Streamlit 主界面（判别 / 批量 / 报告）
├── pages/
│   └── 1_使用手册.py              # 使用手册独立页面
├── requirements.txt
└── README.md
```

**数据流概览：**

```
online_retail.csv → 清洗 → RFM → 标准化 → GMM(EM) → output/
                                              ↓
                                    app.py（判别 / 报告）
```

---

## 一键启动（推荐）

完成 [环境准备](#1-环境准备) 后，在项目根目录双击或执行 `run_dev.bat`，将**自动优先使用 `.venv\Scripts\python.exe`**，顺序完成训练并启动 Streamlit：

```powershell
.\run_dev.bat
```

| 命令 | 说明 |
|------|------|
| `run_dev.bat` | 训练四个统计周期模型 → 启动 Streamlit |
| `run_dev.bat --skip-train` | 跳过训练，仅启动应用（需已有 `output/`） |
| `run_dev.bat --train-only` | 仅训练，不打开浏览器 |
| `run_dev.bat --download` | 先下载 UCI 数据，再训练并启动应用 |
| `run_dev.bat --download-only` | 仅下载数据 |
| `run_dev.bat -- --k 4` | `--` 后的参数传给 `run_train.py` |

训练结束后脚本会拉起 Streamlit，浏览器访问提示地址（通常为 `http://localhost:8501`）。在命令行窗口按 `Ctrl+C` 结束服务。

**使用手册：** 启动后在 Streamlit 侧栏进入 **「使用手册」** 页面，查看群体编号对照、RFM 说明与常见问题；主界面仅保留判别操作与核心结果。

---

## 数据来源

### 主数据集：UCI Online Retail

- **名称**：Online Retail Data Set
- **内容**：英国在线零售商 2010-12-01 至 2011-12-09 的交易记录（约 54 万行）
- **主要字段**：`InvoiceNo`、`StockCode`、`Description`、`Quantity`、`InvoiceDate`、`UnitPrice`、`CustomerID`、`Country`
- **用途**：按 `CustomerID` 聚合生成 RFM，作为 GMM 训练特征

**获取方式（二选一）：**

1. **脚本下载（推荐）**

   ```powershell
   python scripts/download_data.py
   ```

   脚本从 UCI 下载 `Online Retail.xlsx` 并转换为 `data/online_retail.csv`。

2. **手动下载**

   从 [UCI Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail) 下载 Excel，另存为 CSV 后放入：

   ```
   data/online_retail.csv
   ```

### 预处理规则（与 `core/dataset.py` 一致）

- 删除 `CustomerID` 为空
- 删除 `Quantity ≤ 0`、`UnitPrice ≤ 0`
- 删除退款单（`InvoiceNo` 以 `C` 开头）
- **Recency**：距数据最大日期次日为快照日的天数
- **Frequency**：独立订单数（`InvoiceNo` 去重）
- **Monetary**：`Quantity × UnitPrice` 求和

### 测试数据

`data/rfm_test_data_100.csv` 可用于批量判别功能联调，无需重新训练。

---

## 本地训练流程（分析人员）

### 1. 环境准备

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\activate    # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备数据

确保存在 `data/online_retail.csv`（见上一节）。

### 3. 执行训练

```powershell
python scripts/run_train.py
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/online_retail.csv` | 原始交易 CSV |
| `--k` | `3` | 聚类数 K |
| `--seed` | `42` | 随机种子 |
| `--output` | `output/` | 产物目录 |

示例：

```powershell
python scripts/run_train.py --data data/online_retail.csv --k 3 --seed 42
```

### 4. 训练阶段说明

`scripts/run_train.py` 对 **四个统计周期** 分别训练（以数据截止日前推窗口截取交易）：

| 目录 | 统计周期 | 窗口天数 |
|------|----------|----------|
| `output/1y/` | 近 1 年 | 365 |
| `output/6m/` | 近半年 | 182 |
| `output/3m/` | 近 1 季度 | 91 |
| `output/1m/` | 近 1 个月 | 30 |

每个周期独立完成：截取交易 → RFM → 标准化 → GMM → 业务标签 → PCA → 写入对应子目录，并生成 `output/manifest.json`。

### 5. 训练产物

每个 `output/{周期}/` 下含：`gmm_model.pkl`、`model_meta.json`、`user_segments.csv`、`cluster_summary.csv`、`pca_points.csv`。

**更新数据后**：重新执行 `python scripts/run_train.py`，客户刷新网页并选择周期即可。

---

## 客户端使用流程（业务人员）

### 1. 启动应用

需先完成训练，且 `output/` 中存在 `gmm_model.pkl` 等文件。

```powershell
streamlit run app.py
```

也可直接运行（会自动拉起 Streamlit）：

```powershell
python app.py
```

浏览器打开提示的本地地址（通常为 `http://localhost:8501`）。

### 2. 功能说明

侧栏选择 **统计周期**，主界面三个标签页：**用户判别**、**批量判别**、**整体报告**（展示最近一次批量判别的汇总与图表，非训练集）。指标含义、群体编号对照见 **「使用手册」** 页面。

#### （1）用户判别

- **输入**：最近消费天数、购买次数、消费金额（可选用户编号）
- **说明**：切换周期会加载对应模型；购买次数/金额须按所选窗口汇总（界面不展示历史具体日期）
- **输出**：所属群体、置信度、个性化解读、群体平均画像、运营建议、各群体归属概率条形图

#### （2）批量判别

- **输入**：含 `Recency`、`Frequency`、`Monetary` 的 CSV（可选 `CustomerID`）
- **输出**：逐行分群结果，可下载 `batch_predict_result.csv`
- 页面提供模板下载

#### （3）整体报告

- 须先在「批量判别」完成一次上传判别
- 展示**本批 CSV** 的用户数、各群体汇总、明细表及分群分布/规模/RFM 图
- 可下载本批判别结果 CSV

### 3. 客户无需了解的内容

- 聚类数 K、EM 迭代、BIC/AIC 等建模细节
- 只需准备符合定义的 RFM 指标即可获得判断

### 4. 交付建议

1. 分析侧先跑通 `python scripts/run_train.py`，确认 `output/` 完整
2. 将项目（至少含 `output/`、`app.py`、`core/`、`ui/`、`utils/`、`requirements.txt`）部署到客户环境
3. 约定：**整体报告**反映业务方上传批次的判别结果；模型更新后由分析团队重新训练并替换 `output/` 中的 `gmm_model.pkl`

### 5. 常见问题

| 现象 | 处理 |
|------|------|
| 提示「未找到已训练模型」 | 执行 `python scripts/run_train.py` |
| 判别结果与「流失」字面不符 | 标签描述的是 **群体平均画像**；单用户解读见结果中的「针对您本次输入的解读」 |
| 群体编号在不同周期是否一致 | **是**：第 1/2/3 类与三类名称全项目固定；仅底层 GMM 簇下标随周期变化，见使用手册中的「本周期模型映射」 |
| 输入 RFM 周期与训练数据不同 | 允许；模型在标准化空间内比较，详见使用手册中的训练样本参考范围 |
| 整体报告为空 | 请先在「批量判别」页完成一次上传判别 |

---

## 仓库与首次克隆

- GitHub：[YF-Yan/Data_Mining](https://github.com/YF-Yan/Data_Mining)
- 克隆后需自行准备数据并训练（仓库不含大体积 CSV 与 `gmm_model.pkl`）：

```powershell
python scripts/download_data.py
python scripts/run_train.py
streamlit run app.py
```

---

## 项目参考资料

### 数据集与业务方法

- [UCI Machine Learning Repository: Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail) — 本项目主数据源
- RFM 分析：Recency / Frequency / Monetary 用户价值分层经典框架

### 算法与实现

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm.* — EM 算法原始论文
- [scikit-learn: GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) — 本项目 GMM/EM 实现
- [scikit-learn: silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) — 轮廓系数
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — 报告可视化降维

### 应用框架

- [Streamlit Documentation](https://docs.streamlit.io/) — 客户端 Web 界面

### 依赖版本（见 `requirements.txt`）

- pandas ≥ 2.0、numpy ≥ 1.24、scikit-learn ≥ 1.3、matplotlib ≥ 3.7、streamlit ≥ 1.28、joblib ≥ 1.3、openpyxl ≥ 3.1（数据下载脚本读 xlsx 用）

---

## 快速命令索引

```powershell
# 一键：训练 + 启动客户端（推荐）
.\run_dev.bat

# 分步执行
python scripts/download_data.py
python scripts/run_train.py
streamlit run app.py
# 或
python app.py
```

---

*课程/数据挖掘项目：基于 EM 算法的用户行为分群与分析*
