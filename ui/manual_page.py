"""
使用手册独立页面内容。
"""

from typing import Dict, List, Optional

import streamlit as st

from ui.components import (
    get_period_label,
    render_period_cluster_mapping,
    render_segment_catalog_table,
    training_period_note,
)
from utils.io import load_segment_catalog


def render_manual_page(meta: dict, manifest: Optional[dict] = None) -> None:
    catalog = load_segment_catalog(period_key=meta.get("period_key"))
    period_label = get_period_label(meta)

    st.markdown(
        """
        本页说明如何填写 RFM、如何理解判别结果，以及**群体编号与所属群体**的固定对应关系。
        日常操作请在首页完成判别。
        """
    )

    st.header("1. 群体编号（全周期固定）")
    st.markdown(
        "**第 1 / 2 / 3 类** 与下表名称在全项目内一一对应；切换统计周期时编号与名称不变。"
    )
    render_segment_catalog_table(catalog)

    st.header("2. 本周期模型映射")
    st.caption(
        f"当前侧边栏所选周期：**{period_label}**。下表为本次训练中 GMM 簇下标与统一编号的对应。"
    )
    render_period_cluster_mapping(meta)

    st.header("3. RFM 怎么填")
    st.markdown(
        f"""
| 指标 | 含义 |
|------|------|
| **最近消费（天）** | 距分析截止日，上次购买过了几天；**越小越近** |
| **购买次数** | 在 **{period_label}** 内的订单笔数（按订单号去重） |
| **消费金额** | 在 **{period_label}** 内的总花费（数量×单价） |

{training_period_note(meta)}

填写须与所选**统计周期**一致；周期在侧边栏切换后会自动换用对应模型。
        """
    )

    bounds = meta.get("rfm_bounds") or {}
    if bounds:
        r_b, f_b, m_b = bounds.get("recency", {}), bounds.get("frequency", {}), bounds.get("monetary", {})
        st.markdown("**本周期训练样本参考范围**")
        st.markdown(
            f"- 最近消费：{r_b.get('min', 0):.0f}～{r_b.get('max', 0):.0f} 天（中位数约 {r_b.get('median', 0):.0f}）  \n"
            f"- 购买次数：{f_b.get('min', 0):.0f}～{f_b.get('max', 0):.0f}（中位数约 {f_b.get('median', 0):.0f}）  \n"
            f"- 消费金额：{m_b.get('min', 0):.0f}～{m_b.get('max', 0):.0f}（中位数约 {m_b.get('median', 0):.0f}）  \n"
            "超出范围的输入会被模型裁剪处理，极端值可能影响分群稳定性。"
        )

    st.header("4. 结果怎么读")
    st.markdown(
        """
| 字段 | 说明 |
|------|------|
| **所属群体** | 统一编号对应的业务名称（见第 1 节） |
| **群体编号** | 第 1 / 2 / 3 类，与所属群体固定配对 |
| **模型置信度** | 模型认为属于该类的概率，高不代表业务上一定「典型」 |
| **针对您本次输入的解读** | 结合您填写的 R、F、M 的说明，**优先参考** |
| **运营建议** | 按统一群体编号给出的策略提示 |

**注意：** 「所属群体」描述该类用户的**平均画像**；若与您的输入字面不符，以「针对您本次输入的解读」为准。
        """
    )

    st.header("5. 批量判别与整体报告")
    st.markdown(
        """
1. 在 **「批量判别」** 上传 CSV（列：`Recency`、`Frequency`、`Monetary`，可选 `CustomerID`）
2. 点击 **批量判别** 后，切换到 **「整体报告」** 查看本批用户的群体汇总、图表与明细
3. `Frequency`、`Monetary` 须按当前统计周期汇总

整体报告展示的是**您上传批次的判别结果**，不是模型训练集统计。
        """
    )

    st.header("6. 常见问题")
    st.markdown(
        """
**Q：为什么最近刚买过，却显示「流失风险」？**  
A：标签来自该类用户的平均行为；您可能频次/金额更接近该类。请看结果中的个性化解读。

**Q：换统计周期结果会变吗？**  
A：会。不同周期使用独立训练的模型，RFM 也须按对应窗口重新汇总。

**Q：群体编号和所属群体会错位吗？**  
A：不会。第 1 类始终是核心高价值，第 2 类流失风险，第 3 类活跃潜力；仅底层簇下标随周期变化（见第 2 节）。
        """
    )

    with st.expander("管理员：训练与部署"):
        st.markdown(
            """
1. 数据放入 `data/online_retail.csv`
2. 执行 `python scripts/run_train.py` 或 `run_dev.bat`
3. 启动 `streamlit run app.py`

数据来源：UCI Online Retail；算法为 RFM + GMM（EM）。
            """
        )
