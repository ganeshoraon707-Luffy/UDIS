"""
dashboard.py
UDIS - University Decision Intelligence System
Editorial light theme dashboard.
Run: streamlit run dashboard/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import json, glob
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from etl import get_clean_data
from predict import predict_stage1, predict_stage2, predict_stage3, predict_stage4, get_model_path

st.set_page_config(page_title="UDIS", page_icon="U", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Serif+Display&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#f5f2ee !important;font-family:'DM Sans',sans-serif}
.main .block-container{padding:0 !important;max-width:100% !important}
[data-testid="stAppViewContainer"]>.main>div{padding:0 !important;max-width:100% !important}
[data-testid="stSidebar"],[data-testid="collapsedControl"],header,footer{display:none !important}
div[data-testid="stVerticalBlock"]>div{padding-top:0 !important;padding-bottom:0 !important}
.stSlider label{color:#888 !important;font-size:10px !important;text-transform:uppercase;letter-spacing:0.8px;font-weight:700 !important}
div[data-testid="stSlider"]>div>div{background:#e0ddd8 !important}
div[data-testid="stSlider"]>div>div>div{background:#1a1a1a !important}
button[kind="primary"]{background:#1a1a1a !important;color:#f5f2ee !important;border:none !important;
  border-radius:4px !important;font-family:'DM Sans',sans-serif !important;font-weight:700 !important;
  text-transform:uppercase !important;letter-spacing:1px !important;width:100% !important;font-size:11px !important}
</style>""", unsafe_allow_html=True)

ROOT = os.path.join(os.path.dirname(__file__), "..")

@st.cache_data
def load_data():
    return get_clean_data(os.path.join(ROOT, "data", "university.db"))

@st.cache_data
def load_meta():
    files = sorted(glob.glob(os.path.join(ROOT,"models","metadata_stage*.json")),
        key=lambda p: int(p.split("stage")[1].replace(".json","")))
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT,"models","metadata_v*.json")),
            key=lambda p: int(p.split("_v")[1].replace(".json","")))
    if files:
        with open(files[-1]) as f:
            return json.load(f)
    return {}

df   = load_data()
meta = load_meta()
rc   = df["risk"].value_counts().to_dict()
n_h  = rc.get("High", 0)
n_m  = rc.get("Medium", 0)
n_l  = rc.get("Low", 0)
total = len(df)
best_name = meta.get("best_model", "LR")
best_f1   = meta.get("macro_f1", 0.877)
best_acc  = meta.get("accuracy", 0.877)
att_avg   = round(df[["ut1_attendance","mid_attendance","ut2_attendance","end_attendance"]].mean().mean(), 1)
df["acad_pct"] = (0.125*(df["ut1_marks"]/10*100) + 0.25*(df["mid_marks"]/20*100) +
                  0.125*(df["ut2_marks"]/10*100) + 0.5*(df["end_marks"]/40*100))
marks_avg = round(df["acad_pct"].mean(), 1)

# ── TOP BAR ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:center;justify-content:space-between;'
    'padding:0 32px;height:58px;background:#1a1a1a;border-bottom:2px solid #e8c547">'
    '<div>'
    '<span style="font-family:DM Serif Display,serif;font-size:22px;color:#f5f2ee;letter-spacing:-0.5px">'
    'UD<span style="color:#e8c547">I</span>S</span>'
    '<span style="font-size:10px;letter-spacing:2px;color:#555;text-transform:uppercase;margin-left:12px">'
    'University Decision Intelligence System</span>'
    '</div>'
    '<div style="display:flex;gap:28px">'
    + "".join([
        '<div style="text-align:right"><div style="font-size:18px;font-weight:700;color:#f5f2ee;line-height:1">{v}</div>'
        '<div style="font-size:9px;text-transform:uppercase;letter-spacing:1px;color:#555">{l}</div></div>'.format(v=v, l=l)
        for v, l in [
            (f"{total:,}", "Records"),
            ("24", "Departments"),
            ("4", "ML Models"),
            (f"{int(best_acc*100)}%", "Accuracy"),
        ]
    ])
    + '</div></div>',
    unsafe_allow_html=True
)

# ── ROW 1: 4 stat cells ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1,1,1,1])

with c1:
    st.markdown(
        '<div style="background:#1a1a1a;padding:22px;height:200px;display:flex;flex-direction:column;'
        'justify-content:space-between;border-right:1px solid #333;border-bottom:1px solid #333">'
        '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#555">'
        'Best Model — Stage 4</div>'
        '<div>'
        '<div style="font-family:DM Serif Display,serif;font-size:30px;color:#ffffff;line-height:1.1;margin-bottom:5px">'
        + best_name +
        '</div>'
        '<div style="font-size:12px;color:#666">Macro F1 — ' + str(best_f1) + '</div>'
        '</div>'
        '<div style="display:flex;gap:6px">'
        + "".join([
            '<div style="background:#2a2a2a;border-radius:3px;padding:4px 8px;font-size:9px;'
            'color:#888;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">' + m + '</div>'
            for m in ["SVM","LR","RF","GB"]
        ])
        + '</div></div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        '<div style="background:#e8c547;padding:22px;height:200px;display:flex;flex-direction:column;'
        'justify-content:space-between;border-right:1px solid #d4b43e;border-bottom:1px solid #d4b43e">'
        '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#8a7520">High Risk Departments</div>'
        '<div>'
        '<div style="font-family:DM Serif Display,serif;font-size:72px;line-height:1;color:#1a1a1a;letter-spacing:-3px">'
        + str(n_h) +
        '</div>'
        '<div style="font-size:12px;color:#8a7520;margin-top:4px">' + str(round(n_h/total*100,1)) + '% of all departments</div>'
        '</div>'
        '<div style="font-size:11px;color:#7a6515;font-weight:500">Require immediate intervention</div>'
        '</div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        '<div style="background:#ffffff;padding:22px;height:200px;border-right:1px solid #ddd8d0;border-bottom:1px solid #ddd8d0">'
        '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#aaa;margin-bottom:16px">Dataset Overview</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
        + "".join([
            '<div><div style="font-family:DM Serif Display,serif;font-size:32px;color:#1a1a1a;line-height:1">{v}</div>'
            '<div style="font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:0.8px;margin-top:3px">{l}</div></div>'.format(v=v, l=l)
            for v, l in [
                (n_m, "Medium Risk"),
                (n_l, "Low Risk"),
                (f"{att_avg}%", "Avg Attendance"),
                (f"{marks_avg}%", "Avg Marks"),
            ]
        ])
        + '</div></div>',
        unsafe_allow_html=True
    )

with c4:
    model_colors = {"Logistic Regression":"#1a1a1a","Random Forest":"#e8c547",
                    "Gradient Boosting":"#c8c8c8","SVM":"#888888"}
    rows = ""
    all_results = meta.get("all_results", {})
    for name, res in all_results.items():
        f1   = res.get("macro_f1", 0)
        pct  = int(f1 * 100)
        col  = model_colors.get(name, "#888")
        tag  = " ★" if name == best_name else ""
        short = (name.replace("Logistic Regression","LR")
                     .replace("Random Forest","RF")
                     .replace("Gradient Boosting","GB"))
        rows += (
            '<div style="display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #f0ece8">'
            '<span style="font-size:11px;color:#888;min-width:65px">' + short + tag + '</span>'
            '<div style="flex:1;height:4px;background:#e8e4df;border-radius:2px;overflow:hidden">'
            '<div style="width:' + str(pct) + '%;height:100%;background:' + col + ';border-radius:2px"></div>'
            '</div>'
            '<span style="font-size:11px;font-weight:700;color:#1a1a1a;min-width:36px;text-align:right">' + str(f1) + '</span>'
            '</div>'
        )
    st.markdown(
        '<div style="background:#f5f2ee;padding:22px;height:200px;border-left:0.5px solid #ddd8d0;border-bottom:1px solid #ddd8d0">'
        '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#aaa;margin-bottom:12px">Model Comparison</div>'
        + rows + '</div>',
        unsafe_allow_html=True
    )

# ── ROW 2: Charts ───────────────────────────────────────────────────────────────
ch1, ch2, ch3 = st.columns([2,1,1])

with ch1:
    st.markdown(
        '<div style="padding:14px 22px 4px;border-top:1px solid #ddd8d0;border-right:1px solid #ddd8d0;background:#fff">'
        '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#aaa">Academic Score by Department — Top 10 (black) vs Bottom 3 (gold)</div>'
        '</div>',
        unsafe_allow_html=True
    )
    dept_scores = df.groupby("department")["acad_pct"].mean().reset_index()
    dept_scores.columns = ["Dept","Score"]
    dept_scores = dept_scores.sort_values("Score", ascending=False)
    combined = pd.concat([dept_scores.head(10), dept_scores.tail(3)]).reset_index(drop=True)
    bar_colors = ["#1a1a1a"]*10 + ["#e8c547"]*3
    fig_d = go.Figure(go.Bar(
        x=combined["Dept"], y=combined["Score"].round(1),
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:.0f}%" for v in combined["Score"]],
        textposition="outside", textfont=dict(size=8, color="#888"),
    ))
    fig_d.add_hline(y=70, line_dash="dash", line_color="#2e7d32", line_width=1,
        annotation_text="Low Risk (70%)", annotation_font=dict(size=8, color="#2e7d32"), annotation_position="right")
    fig_d.add_hline(y=50, line_dash="dash", line_color="#c0392b", line_width=1,
        annotation_text="High Risk (50%)", annotation_font=dict(size=8, color="#c0392b"), annotation_position="right")
    fig_d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        margin=dict(l=0,r=70,t=4,b=0), height=190,
        xaxis=dict(color="#ccc", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=8,color="#aaa"), tickangle=-35),
        yaxis=dict(color="#ccc", gridcolor="rgba(200,200,200,0.4)", tickfont=dict(size=9,color="#aaa"),
                   range=[0,110], ticksuffix="%"),
        showlegend=False, bargap=0.35
    )
    st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

with ch2:
    st.markdown(
        '<div style="padding:14px 22px 4px;border-top:1px solid #ddd8d0;border-right:1px solid #ddd8d0;background:#f5f2ee">'
        '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#aaa">Attendance Trend (%)</div>'
        '</div>',
        unsafe_allow_html=True
    )
    avg_att_vals = [round(v,1) for v in df[["ut1_attendance","mid_attendance","ut2_attendance","end_attendance"]].mean().values]
    fig_a = go.Figure()
    fig_a.add_hline(y=75, line_dash="dash", line_color="#e8c547", line_width=1.5,
        annotation_text="75% min", annotation_font=dict(size=8,color="#8a7520"), annotation_position="right")
    fig_a.add_hrect(y0=0, y1=75, fillcolor="rgba(192,57,43,0.04)", line_width=0)
    fig_a.add_trace(go.Scatter(
        x=["UT1","Mid","UT2","End"], y=avg_att_vals,
        mode="lines+markers+text",
        line=dict(color="#1a1a1a", width=2),
        marker=dict(size=8, color="#1a1a1a"),
        text=[f"{v}%" for v in avg_att_vals],
        textposition="top center",
        textfont=dict(size=8, color="#888"),
        fill="tozeroy", fillcolor="rgba(26,26,26,0.04)"
    ))
    fig_a.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(245,242,238,1)",
        margin=dict(l=0,r=40,t=4,b=0), height=190,
        xaxis=dict(color="#ccc", gridcolor="rgba(200,200,200,0.4)", tickfont=dict(size=9,color="#aaa")),
        yaxis=dict(color="#ccc", gridcolor="rgba(200,200,200,0.4)", tickfont=dict(size=9,color="#aaa"),
                   range=[0,105], dtick=25, ticksuffix="%"),
        showlegend=False
    )
    st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})

with ch3:
    st.markdown(
        '<div style="padding:14px 22px 4px;border-top:1px solid #ddd8d0;background:#f5f2ee">'
        '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#aaa">Risk Distribution</div>'
        '</div>',
        unsafe_allow_html=True
    )
    rc_df = df["risk"].value_counts().reset_index()
    rc_df.columns = ["Risk","Count"]
    cmap = {"High":"#c0392b","Medium":"#e8c547","Low":"#2e7d32"}
    fig_p = go.Figure(go.Pie(
        labels=rc_df["Risk"], values=rc_df["Count"], hole=0.55,
        marker=dict(colors=[cmap.get(r,"#888") for r in rc_df["Risk"]],
                    line=dict(color="#f5f2ee", width=3)),
        textfont=dict(size=10), textinfo="percent"
    ))
    fig_p.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=4,b=0), height=190,
        showlegend=True,
        legend=dict(font=dict(size=10,color="#888"), bgcolor="rgba(0,0,0,0)", orientation="v", x=1, y=0.5)
    )
    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

# ── ROW 3: PREDICTION ────────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:2px solid #1a1a1a;background:#ffffff;padding:12px 24px 4px">'
    '<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#aaa">'
    'Risk Prediction — Progressive Monitoring (4 Stage Models)</div>'
    '</div>',
    unsafe_allow_html=True
)

def show_result(risk, acc):
    cls = str(risk)
    advice = {
        "High":   "Immediate intervention needed. Assign mentors and schedule remedial sessions.",
        "Medium": "Monitor closely. Offer doubt-clearing sessions and peer study groups.",
        "Low":    "On track. Maintain current teaching strategy."
    }
    bg   = {"High":"#fce8e8","Medium":"#fef6e0","Low":"#e8f5e8"}
    bdr  = {"High":"#f0a0a0","Medium":"#e8c547","Low":"#90c090"}
    fcol = {"High":"#c0392b","Medium":"#8a7520","Low":"#2e7d32"}
    st.markdown(
        '<div style="background:' + bg[cls] + ';border:1px solid ' + bdr[cls] + ';border-radius:5px;padding:10px 13px;margin-top:10px">'
        '<div style="font-family:DM Serif Display,serif;font-size:18px;color:' + fcol[cls] + ';margin-bottom:4px">' + cls + ' Risk</div>'
        '<div style="font-size:11px;color:#555;line-height:1.5">' + advice[cls] + '</div>'
        '<div style="font-size:9px;color:#aaa;margin-top:5px;text-transform:uppercase;letter-spacing:0.8px">Stage model accuracy: ' + acc + '</div>'
        '</div>',
        unsafe_allow_html=True
    )

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown(
        '<div style="border-right:1px solid #ddd8d0;padding:4px 8px 0">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        'color:#1a1a1a;margin-bottom:10px;border-bottom:2px solid #e8c547;padding-bottom:6px">'
        'Stage 1 — After UT1</div></div>',
        unsafe_allow_html=True
    )
    ut1m  = st.slider("UT1 marks (out of 10)", 0, 10, 7, key="s1m")
    ut1a  = st.slider("UT1 attendance %", 35, 100, 82, key="s1a")
    if st.button("Predict →", type="primary", key="pb1"):
        show_result(str(predict_stage1(ut1m, ut1a)), "78%")

with p2:
    st.markdown(
        '<div style="border-right:1px solid #ddd8d0;padding:4px 8px 0">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        'color:#1a1a1a;margin-bottom:10px;border-bottom:2px solid #e8c547;padding-bottom:6px">'
        'Stage 2 — After Mid Exam</div></div>',
        unsafe_allow_html=True
    )
    s2a, s2b = st.columns(2)
    with s2a:
        s2m1 = st.slider("UT1 /10", 0, 10, 7, key="s2m1")
        s2m2 = st.slider("Mid /20", 0, 20, 14, key="s2m2")
    with s2b:
        s2a1 = st.slider("UT1 att%", 35, 100, 82, key="s2a1")
        s2a2 = st.slider("Mid att%", 35, 100, 79, key="s2a2")
    if st.button("Predict →", type="primary", key="pb2"):
        show_result(str(predict_stage2(s2m1, s2m2, s2a1, s2a2)), "81%")

with p3:
    st.markdown(
        '<div style="border-right:1px solid #ddd8d0;padding:4px 8px 0">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        'color:#1a1a1a;margin-bottom:10px;border-bottom:2px solid #e8c547;padding-bottom:6px">'
        'Stage 3 — After UT2</div></div>',
        unsafe_allow_html=True
    )
    s3a, s3b = st.columns(2)
    with s3a:
        s3m1 = st.slider("UT1 /10", 0, 10, 7, key="s3m1")
        s3m2 = st.slider("Mid /20", 0, 20, 14, key="s3m2")
        s3m3 = st.slider("UT2 /10", 0, 10, 7, key="s3m3")
    with s3b:
        s3a1 = st.slider("UT1 att%", 35, 100, 82, key="s3a1")
        s3a2 = st.slider("Mid att%", 35, 100, 79, key="s3a2")
        s3a3 = st.slider("UT2 att%", 35, 100, 76, key="s3a3")
    if st.button("Predict →", type="primary", key="pb3"):
        show_result(str(predict_stage3(s3m1, s3m2, s3m3, s3a1, s3a2, s3a3)), "83%")

with p4:
    st.markdown(
        '<div style="padding:4px 8px 0">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        'color:#1a1a1a;margin-bottom:10px;border-bottom:2px solid #1a1a1a;padding-bottom:6px">'
        'Stage 4 — Full Semester</div></div>',
        unsafe_allow_html=True
    )
    s4a, s4b = st.columns(2)
    with s4a:
        s4m1 = st.slider("UT1 /10", 0, 10, 7, key="s4m1")
        s4m2 = st.slider("Mid /20", 0, 20, 14, key="s4m2")
        s4m3 = st.slider("UT2 /10", 0, 10, 7, key="s4m3")
        s4m4 = st.slider("End /40", 0, 40, 28, key="s4m4")
    with s4b:
        s4a1 = st.slider("UT1 att%", 35, 100, 82, key="s4a1")
        s4a2 = st.slider("Mid att%", 35, 100, 79, key="s4a2")
        s4a3 = st.slider("UT2 att%", 35, 100, 76, key="s4a3")
        s4a4 = st.slider("End att%", 35, 100, 73, key="s4a4")
    if st.button("Predict →", type="primary", key="pb4"):
        show_result(str(predict_stage4(s4m1,s4m2,s4m3,s4m4,s4a1,s4a2,s4a3,s4a4)), "87%")
