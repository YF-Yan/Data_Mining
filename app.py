"""
用户行为分群 — 客户端主界面（输入 RFM → 分群判断）

启动：streamlit run app.py
使用手册：侧栏「使用手册」页面
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.predictor import predict_batch, predict_rfm
from core.segments import resolve_segment_display
from ui.batch_report import render_batch_report_tab, save_batch_report
from ui.components import get_period_label, label_frequency, label_monetary
from ui.layout import init_app_context
from utils.io import artifacts_ready


def _in_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def render_predict_tab(bundle: dict, meta: dict):
    period_label = get_period_label(meta)
    st.caption(f"当前周期：**{period_label}** · 说明见「使用手册」")

    bounds = meta.get("rfm_bounds", {})
    r_b = bounds.get("recency", {})
    f_b = bounds.get("frequency", {})
    m_b = bounds.get("monetary", {})

    with st.form("rfm_input_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.number_input(
                "最近消费 (天)",
                min_value=0.0,
                value=float(r_b.get("median", 50)),
                step=1.0,
            )
        with c2:
            frequency = st.number_input(
                label_frequency(meta),
                min_value=0.0,
                value=float(f_b.get("median", 5)),
                step=1.0,
            )
        with c3:
            monetary = st.number_input(
                label_monetary(meta),
                min_value=0.0,
                value=float(m_b.get("median", 1000)),
                step=100.0,
                format="%.2f",
            )
        customer_id = st.text_input("用户编号（可选）")
        submitted = st.form_submit_button("开始判别", type="primary", use_container_width=True)

    if not submitted:
        return

    try:
        result = predict_rfm(
            recency=recency,
            frequency=frequency,
            monetary=monetary,
            gmm=bundle["gmm"],
            scaler=bundle["scaler"],
            feature_cols=bundle["feature_cols"],
            segment_profiles=meta["segment_profiles"],
            rfm_bounds=meta.get("rfm_bounds"),
            clip_bounds=bundle.get("clip_bounds"),
        )
    except ValueError as e:
        st.error(str(e))
        return

    if customer_id.strip():
        st.caption(f"用户编号：{customer_id.strip()}")

    segment_id = result.get("segment_id", result["cluster"] + 1)
    m1, m2, m3 = st.columns(3)
    m1.metric("所属群体", result["segment_name"])
    m2.metric("群体编号", f"第 {segment_id} 类")
    m3.metric("置信度", f"{result['confidence'] * 100:.1f}%")

    st.info(result.get("personalized_insight", ""))
    st.warning(f"运营建议：{result['advice']}")

    profiles = meta["segment_profiles"]
    prob_rows = []
    for cid, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        disp = resolve_segment_display(int(cid), profiles)
        prob_rows.append({"群体": disp["segment_name"], "概率": prob})
    prob_df = pd.DataFrame(prob_rows)
    st.bar_chart(prob_df.set_index("群体")["概率"], height=220)


def render_batch_tab(bundle: dict, meta: dict, period_key: str):
    period_label = get_period_label(meta)
    st.caption(f"当前周期：**{period_label}** · 完成后到「整体报告」查看汇总")

    template = pd.DataFrame(
        [
            {"CustomerID": 10001, "Recency": 15, "Frequency": 12, "Monetary": 5200.5},
            {"CustomerID": 10002, "Recency": 120, "Frequency": 2, "Monetary": 350.0},
        ]
    )
    st.download_button(
        "下载模板",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="batch_rfm_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("上传 CSV", type=["csv"], label_visibility="collapsed")
    if not st.button("批量判别", type="primary", disabled=uploaded is None):
        return

    try:
        raw = pd.read_csv(uploaded, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(uploaded, encoding="latin-1")

    try:
        batch_result = predict_batch(
            raw,
            gmm=bundle["gmm"],
            scaler=bundle["scaler"],
            feature_cols=bundle["feature_cols"],
            segment_profiles=meta["segment_profiles"],
            clip_bounds=bundle.get("clip_bounds"),
        )
    except ValueError as e:
        st.error(str(e))
        return

    save_batch_report(
        period_key,
        period_label,
        batch_result,
        file_name=getattr(uploaded, "name", ""),
    )
    st.success(f"已完成 {len(batch_result)} 条，请切换到 **「整体报告」** 查看图表与汇总。")


def main():
    st.set_page_config(page_title="用户分群判别", page_icon="🎯", layout="wide")
    st.title("用户分群判别")

    if not artifacts_ready():
        st.error("未找到已训练模型，请先执行 `python scripts/run_train.py`。")
        return

    period_key, bundle, meta, _manifest = init_app_context()
    if meta is None:
        return

    period_label = get_period_label(meta)
    tab_predict, tab_batch, tab_report = st.tabs(["用户判别", "批量判别", "整体报告"])

    with tab_predict:
        render_predict_tab(bundle, meta)
    with tab_batch:
        render_batch_tab(bundle, meta, period_key)
    with tab_report:
        render_batch_report_tab(period_key, period_label, bundle)


def _launch_streamlit():
    app_path = str(Path(__file__).resolve())
    print("正在启动 Streamlit 服务...")
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", app_path, *sys.argv[1:]]
        )
    )


if _in_streamlit_runtime():
    main()
elif __name__ == "__main__":
    _launch_streamlit()
