"""
Out-of-School Children and Youth in the Philippines: Regional and Demographic Patterns
SDG 4 (Quality Education), Target 4.1 | Source: PSA 2022 APIS Public-Use File (cleaned)

Structured after "4. Dashboard Requirements" (4.2 minimum content, 4.3 organization):
  Title + stakeholder question -> 1 Context/Problem -> 2 Evidence -> 3 Insight
  -> 4 Implication -> 5 Action/Next step -> Limitations, definitions, sources

Run with:  python -m streamlit run app.py
Expects "APIS PUF 2022 Cleaned.csv" next to this file, or upload it from the sidebar.
"""

import html
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ------------------------------------------------------------------ #
# Setup and constants
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Out-of-School Children and Youth | APIS 2022",
    page_icon="📚",
    layout="wide",
)

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', 'Source Sans Pro', sans-serif; }
.stApp { background: #EDF2F8; color: #1A2230; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1480px; }
h1, h2, h3, h4, h5, p, li, label, .stMarkdown { color: #1A2230; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: #FFFFFF; border-radius: 0 26px 26px 0; }
section[data-testid="stSidebar"] > div { padding-top: .6rem; }
.brand { display:flex; align-items:center; gap:.7rem; margin: .2rem 0 1.1rem 0; }
.logo { width:42px; height:42px; border-radius:50%; background:#1E5AA8; color:#fff;
        display:flex; align-items:center; justify-content:center; font-size:1.25rem; }
.brand-name { font-weight:700; font-size:1rem; line-height:1.15; }
.brand-sub { font-size:.74rem; color:#7A8794; }
.eyebrow { font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; color:#7A8794; font-weight:600; margin:.2rem 0 .5rem 0; }
.nav a { display:block; padding:.5rem .8rem; margin-bottom:2px; border-radius:11px; color:#27303B !important;
         text-decoration:none; font-size:.9rem; font-weight:500; }
.nav a:hover { background:#E3EEFA; color:#1E5AA8 !important; }
div[data-baseweb="select"] > div { background:#F1F5F9; border:none; border-radius:12px; }

/* ---------- top bar ---------- */
.topbar { background:#fff; border-radius:24px; padding:1.1rem 1.5rem; display:flex; align-items:center;
          justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; }
.topbar .title { font-size:1.55rem; font-weight:700; line-height:1.2; }
.topbar .sub { color:#6B7785; font-size:.95rem; margin-top:.15rem; }
.pills { display:flex; gap:.5rem; flex-wrap:wrap; }
.pill { background:#EEF3F9; color:#27303B; border-radius:999px; padding:.38rem .85rem; font-size:.8rem; font-weight:500; }
.pill.dark { background:#10233F; color:#fff; }

/* ---------- section label ---------- */
.sec { display:flex; align-items:center; gap:.65rem; font-size:1.25rem; font-weight:700; margin:1.7rem 0 .8rem 0; }
.sec .num { background:#1E5AA8; color:#fff; width:28px; height:28px; border-radius:50%;
            display:inline-flex; align-items:center; justify-content:center; font-size:.85rem; }

/* ---------- cards ---------- */
div[class*="st-key-card_"] { background:#fff; border-radius:24px; padding:1.15rem 1.4rem 1rem 1.4rem;
                             box-shadow:0 1px 2px rgba(20,35,60,.06); }
div[class*="st-key-card_"] h3 { font-size:1.12rem; font-weight:600; line-height:1.35; margin:0 0 .3rem 0; padding:0; }
div[class*="st-key-card_"] h5 { font-size:.95rem; font-weight:600; margin-top:0; }
div.st-key-card_question { background:#E3EEFA; }

/* ---------- KPI cards ---------- */
.kpi { background:#fff; border-radius:24px; padding:1rem 1.2rem; min-height:128px; box-shadow:0 1px 2px rgba(20,35,60,.06); }
.kpi-top { display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }
.kpi-value { font-size:2.15rem; font-weight:600; line-height:1.15; }
.badge { background:#E3EEFA; color:#1E5AA8; border-radius:999px; padding:.18rem .6rem; font-size:.72rem; font-weight:600; }
.kpi-label { color:#6B7785; font-size:.88rem; margin-top:.5rem; }
.kpi-note { color:#8794A0; font-size:.76rem; margin-top:.15rem; line-height:1.3; }

/* ---------- mini stats, notes ---------- */
.mini-row { display:flex; gap:.7rem; margin:.2rem 0 .5rem 0; flex-wrap:wrap; }
.mini { background:#F1F5F9; border-radius:14px; padding:.5rem .9rem; display:flex; flex-direction:column; }
.mini span { font-size:.74rem; color:#6B7785; }
.mini b { font-size:1.25rem; }
.note { border-radius:16px; padding:.8rem 1.1rem; font-size:.9rem; margin:.9rem 0 .2rem 0; line-height:1.5; }
.note.warn { background:#FFF4DE; }
.note.info { background:#E8F0FB; }
.caption-s { color:#7A8794; font-size:.8rem; line-height:1.45; margin-top:.2rem; }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

TEMPLATE = "plotly_white"
ACCENT, GREY, RED = "#1E5AA8", "#D3DCE6", "#C8553D"
BLUE, PINK = "#6FA8DC", "#E08FAE"  # male / female
W = "Final Population Weights"
RATE = "Weighted non-attendance (%)"
DEFAULT_CSV = Path(__file__).parent / "APIS PUF 2022 Cleaned.csv"
PSA_URL = (
    "https://psa.gov.ph/content/four-out-five-children-aged-5-24-years-were-"
    "attending-school-school-year-2022-to-2023"
)
MIN_SAMPLE = 100  # warn when a selection has fewer unweighted non-attending records

REQUIRED_COLS = [
    "Region", "Sex", "Age as of Last Birthday", "Attendance Status",
    "Age Group", "Why not attending school", W,
]
AGE_ORDER = ["5–9", "10–12", "13–15", "16–18", "19–24"]
YOUNGER_BANDS = ["5–9", "10–12", "13–15"]
SHORT_LABELS = {
    "Finished schooling or finished post secondary/college": "Finished schooling",
    "High cost of education/Financial concern": "Education cost",
    "Lack of personal interest": "Personal interest",
    "Too young to go to school": "Too young",
    "Illness/disability": "Illness/disability",
    "Others, specify": "Other reasons",
    "Due to Covid-19 Pandemic Community Quarantine": "COVID-19 quarantine",
    "Accessibility of school": "School accessibility",
}


_PLOTLY_PARAMS = inspect.signature(st.plotly_chart).parameters


def show(fig):
    """Render a Plotly figure full-width on both older and newer Streamlit versions."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Source Sans Pro, sans-serif", size=13, color="#27303B"),
    )
    fig.update_xaxes(gridcolor="#E6ECF3")
    fig.update_yaxes(gridcolor="#E6ECF3")
    if "width" in _PLOTLY_PARAMS:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def card(name: str):
    """A white rounded card (styled via the st-key-card_* CSS class)."""
    return st.container(key=f"card_{name}")


def kpi(label, value, badge=None, note=None, tip=""):
    b = f'<span class="badge">{html.escape(str(badge))}</span>' if badge else ""
    n = f'<div class="kpi-note">{html.escape(str(note))}</div>' if note else ""
    st.markdown(
        f'<div class="kpi" title="{html.escape(str(tip), quote=True)}">'
        f'<div class="kpi-top"><span class="kpi-value">{html.escape(str(value))}</span>{b}</div>'
        f'<div class="kpi-label">{html.escape(str(label))}</div>{n}</div>',
        unsafe_allow_html=True,
    )


def section(num, title, anchor):
    st.markdown(
        f'<div id="{anchor}" class="sec"><span class="num">{num}</span>'
        f"<span>{html.escape(title)}</span></div>",
        unsafe_allow_html=True,
    )


def note_box(text_html: str, kind: str = "warn"):
    st.markdown(f'<div class="note {kind}">{text_html}</div>', unsafe_allow_html=True)


def small(text: str):
    st.markdown(f'<div class="caption-s">{text}</div>', unsafe_allow_html=True)


def short(reason: str) -> str:
    return SHORT_LABELS.get(reason, reason)


# ------------------------------------------------------------------ #
# Data loading and helpers
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner="Loading APIS data...")
def load_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        df["Attendance Status"].isin(["Attending", "Not attending"])
        & df[W].notna()
        & (df[W] > 0)
    ].copy()
    out["Not Attending"] = (out["Attendance Status"] == "Not attending").astype(int)
    out["Sex"] = out["Sex"].astype(str).str.strip()
    return out


def wrate(frame: pd.DataFrame) -> float:
    """Weighted % 'Not attending' = sum(weights of not attending) / sum(weights) * 100."""
    if frame.empty or frame[W].sum() == 0:
        return float("nan")
    return 100 * float((frame["Not Attending"] * frame[W]).sum() / frame[W].sum())


def rate_by(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    num = (frame["Not Attending"] * frame[W]).groupby(frame[col]).sum()
    den = frame[W].groupby(frame[col]).sum()
    return (100 * num / den).rename(RATE).reset_index()


def subset(frame, regions=None, sexes=None, ages=None):
    m = pd.Series(True, index=frame.index)
    if regions is not None:
        m &= frame["Region"].isin(regions)
    if sexes is not None:
        m &= frame["Sex"].isin(sexes)
    if ages is not None:
        m &= frame["Age Group"].isin(ages)
    return frame[m]


def with_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[
        (frame["Attendance Status"] == "Not attending")
        & frame["Why not attending school"].notna()
        & (frame["Why not attending school"].astype(str).str.strip() != "")
    ].copy()
    out["Why not attending school"] = out["Why not attending school"].astype(str).str.strip()
    return out


def reason_shares(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Why not attending school", "Share (%)"])
    return (
        frame.groupby("Why not attending school")[W].sum()
        .div(frame[W].sum()).mul(100)
        .sort_values(ascending=False).rename("Share (%)").reset_index()
    )


def dot_plot(data, cat, order, selected, height, ref_label):
    """Dot plot of weighted rates, sorted highest -> lowest (top to bottom).
    Categories are placed on a numeric axis so the order is exact; selected ones are highlighted."""
    rates = rate_by(data, cat).sort_values(RATE, ascending=False).reset_index(drop=True)
    rates["Pos"] = range(len(rates))
    rates["Label"] = rates[RATE].map(lambda v: f"{v:.1f}%")
    fig = go.Figure()
    for in_sel, name, color in [(True, "In your selection", ACCENT), (False, "Outside selection", GREY)]:
        d = rates[rates[cat].isin(selected) == in_sel]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d[RATE], y=d["Pos"], mode="markers+text", name=name,
            marker=dict(size=15, color=color), text=d["Label"], textposition="middle right",
            customdata=d[[cat]],
            hovertemplate="%{customdata[0]}<br>%{x:.2f}%<extra></extra>",
        ))
    ref = wrate(data)
    if not np.isnan(ref):
        fig.add_vline(x=ref, line_dash="dash", line_color=RED,
                      annotation_text=f"{ref_label}: {ref:.1f}%", annotation_position="top")
    fig.update_xaxes(range=[0, max(rates[RATE].max(), 0 if np.isnan(ref) else ref) * 1.3],
                     title="Weighted non-attendance (% of people in group)")
    fig.update_yaxes(tickmode="array", tickvals=rates["Pos"].tolist(), ticktext=rates[cat].tolist(),
                     autorange="reversed")  # position 0 (highest) at the top
    fig.update_layout(
        template=TEMPLATE, height=height, margin=dict(l=40, r=80, t=60, b=45),
        showlegend=len(selected) < len(order),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig, rates


# ------------------------------------------------------------------ #
# Sidebar: data source
# ------------------------------------------------------------------ #
st.sidebar.markdown(
    '<div class="brand"><div class="logo">📚</div><div><div class="brand-name">APIS 2022</div>'
    '<div class="brand-sub">SDG 4 · Quality Education</div></div></div>'
    '<div class="eyebrow">Story path</div>'
    '<div class="nav">'
    '<a href="#s1">1 · Context</a><a href="#s2">2 · Evidence</a><a href="#s3">3 · Insight</a>'
    '<a href="#s4">4 · Implication</a><a href="#s5">5 · Action</a><a href="#s6">Limitations &amp; sources</a>'
    '</div>',
    unsafe_allow_html=True,
)

raw = None
if DEFAULT_CSV.exists():
    raw = load_csv(DEFAULT_CSV)
else:
    uploaded = st.sidebar.file_uploader("Upload **APIS PUF 2022 Cleaned.csv**", type="csv")
    if uploaded is not None:
        raw = load_csv(uploaded)

if raw is None:
    st.title("Out-of-School Children and Youth in the Philippines")
    st.info("Place **APIS PUF 2022 Cleaned.csv** next to `app.py`, or upload it in the sidebar.")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in raw.columns]
if missing:
    st.error(f"These required columns are missing from the CSV: {missing}")
    st.stop()

nat_df = prepare(raw)  # national, unfiltered (the comparison baseline)

# ------------------------------------------------------------------ #
# Sidebar: filters (purposeful: each answers a stakeholder question)
# ------------------------------------------------------------------ #
all_regions = sorted(nat_df["Region"].unique())
all_sexes = sorted(nat_df["Sex"].unique())
all_ages = [a for a in AGE_ORDER if a in set(nat_df["Age Group"])]

st.sidebar.markdown('<div class="eyebrow" style="margin-top:1.2rem">Filters</div>', unsafe_allow_html=True)
_r = st.sidebar.selectbox(
    "Region", ["All regions"] + all_regions,
    help="Where should we look? Pick one region to highlight it.",
)
_a = st.sidebar.selectbox(
    "Age group", ["All ages"] + all_ages,
    help="Which life stage? Pick one age group to focus on it.",
)
_s = st.sidebar.selectbox(
    "Sex", ["Both"] + all_sexes,
    help="Is the pattern different for males or females?",
)
# "All ..." / "Both" means no restriction
sel_regions = all_regions if _r == "All regions" else [_r]
sel_ages = all_ages if _a == "All ages" else [_a]
sel_sexes = all_sexes if _s == "Both" else [_s]

sel_df = subset(nat_df, sel_regions, sel_sexes, sel_ages)
filters_active = (
    len(sel_regions) < len(all_regions)
    or len(sel_ages) < len(all_ages)
    or len(sel_sexes) < len(all_sexes)
)

# ------------------------------------------------------------------ #
# National baseline numbers (used in framing, implications, claim matrix)
# ------------------------------------------------------------------ #
nat_rate = wrate(nat_df)
nat_regions = rate_by(nat_df, "Region").sort_values(RATE)
nat_hi, nat_lo = nat_regions.iloc[-1], nat_regions.iloc[0]
nat_ages = rate_by(nat_df, "Age Group").set_index("Age Group").reindex(all_ages)[RATE]
nat_top_age = nat_ages.idxmax()
nat_reasons = reason_shares(with_reasons(nat_df))

# ------------------------------------------------------------------ #
# TOP BAR + STAKEHOLDER QUESTION + CORE MESSAGE
# ------------------------------------------------------------------ #
st.markdown(
    '<div class="topbar"><div>'
    '<div class="title">Out-of-School Children and Youth in the Philippines</div>'
    '<div class="sub">Regional and demographic patterns of school non-attendance, ages 5–24</div>'
    '</div><div class="pills"><span class="pill dark">APIS 2022</span>'
    '<span class="pill">SDG 4 · Target 4.1</span><span class="pill">For DepEd &amp; policymakers</span></div></div>',
    unsafe_allow_html=True,
)

q_col, m_col = st.columns([1, 1.3])
with q_col:
    with card("question"):
        st.markdown('<div class="eyebrow">Stakeholder question</div>', unsafe_allow_html=True)
        st.markdown(
            "**DepEd and education policymakers:** Where, and among which groups, were Filipinos "
            "aged 5–24 not attending school in 2022, and what reasons were recorded, so that "
            "attention can be directed to the right places for further investigation?"
        )
with m_col:
    with card("message"):
        st.markdown('<div class="eyebrow">Core message</div>', unsafe_allow_html=True)
        st.markdown(
            f"About **{nat_rate:.1f}%** of people aged 5–24 were not attending school in 2022. "
            f"The estimate is highest for ages **{nat_top_age}** ({nat_ages[nat_top_age]:.1f}%) "
            f"and varies by region, from **{nat_lo[RATE]:.1f}%** ({nat_lo['Region']}) to "
            f"**{nat_hi[RATE]:.1f}%** ({nat_hi['Region']}). Non-attendance reflects different "
            "situations at different ages, so age bands should be read separately and regional "
            "patterns investigated before choosing interventions."
        )

note_box(
    "<b>Read before interpreting.</b> These are weighted descriptive estimates for the population "
    "represented by the <b>2022</b> APIS, not current 2026 figures. “Not attending” does not "
    "automatically mean <i>dropped out</i>; many people aged 19–24 may have finished their "
    "education. Patterns and reasons do not prove cause and effect, and no confidence intervals "
    "are calculated.",
    "warn",
)

# ------------------------------------------------------------------ #
# 1. CONTEXT / PROBLEM
# ------------------------------------------------------------------ #
section("1", "Context: how big is the issue?", "s1")

sel_rate = wrate(sel_df)
n_na = int((sel_df["Not Attending"] == 1).sum())
n_total = len(sel_df)
_kr = rate_by(subset(nat_df, sexes=sel_sexes, ages=sel_ages), "Region").sort_values(RATE)
gap_pts = _kr[RATE].iloc[-1] - _kr[RATE].iloc[0]
gap_hi, gap_lo = _kr.iloc[-1], _kr.iloc[0]

if filters_active:
    small(
        f"Showing your selection: <b>{len(sel_regions)}</b> of {len(all_regions)} regions · "
        f"<b>{len(sel_ages)}</b> of {len(all_ages)} age groups · "
        f"<b>{len(sel_sexes)}</b> of {len(all_sexes)} sex categories. "
        "Differences are measured against the national figure for ages 5–24."
    )

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi(
        "Not attending school", f"{sel_rate:.1f}%",
        badge=f"{sel_rate - nat_rate:+.1f} pts vs national" if filters_active else "National",
        note=f"National: {nat_rate:.1f}%" if filters_active else "Ages 5–24, weighted",
        tip="Weighted share of people in the selection whose attendance status is 'Not attending'.",
    )
with k2:
    kpi("Attending school", f"{100 - sel_rate:.1f}%", note="Ages 5–24, weighted",
        tip="Weighted share of people in the selection who are attending school.")
with k3:
    kpi(
        "Gap between regions", f"{gap_pts:.1f} pts",
        note=f"{gap_hi['Region']} vs {gap_lo['Region']}",
        tip=(
            f"Highest region ({gap_hi['Region']}, {gap_hi[RATE]:.1f}%) minus lowest region "
            f"({gap_lo['Region']}, {gap_lo[RATE]:.1f}%), in percentage points. "
            "Uses the age and sex filters; ignores the region filter."
        ),
    )
with k4:
    kpi("Survey records not attending", f"{n_na:,}", note=f"of {n_total:,} sampled in selection",
        tip="Unweighted number of sampled people not attending.")
if n_na < MIN_SAMPLE:
    note_box(
        f"📉 Only {n_na} sampled people are “not attending” in this selection. Estimates based on "
        "small samples are less stable; interpret with caution.",
        "warn",
    )

# ------------------------------------------------------------------ #
# 2. EVIDENCE
# ------------------------------------------------------------------ #
section("2", "Evidence: where and among whom?", "s2")
left, right = st.columns([1.2, 1])

# --- 2a Regions (ranking / comparison) ---
reg_data = subset(nat_df, sexes=sel_sexes, ages=sel_ages)
reg_rates = rate_by(reg_data, "Region").sort_values(RATE, ascending=False).reset_index(drop=True)
reg_rates["Label"] = reg_rates[RATE].map(lambda v: f"{v:.1f}%")
r_hi, r_lo = reg_rates.iloc[0], reg_rates.iloc[-1]

with left:
    with card("region"):
        st.subheader(
            f"{r_hi['Region']} has the highest non-attendance ({r_hi[RATE]:.1f}%); "
            f"{r_lo['Region']} has the lowest ({r_lo[RATE]:.1f}%)"
        )
        region_filter_on = len(sel_regions) < len(all_regions)
        # Highlight: the region you selected, or (no region filter) the highest region
        focus = set(sel_regions) if region_filter_on else {r_hi["Region"]}
        reg_rates["Pos"] = range(len(reg_rates))  # 0 = highest, drawn at the top
        fig = go.Figure()
        for is_focus, name, color in [
            (True, "In your selection" if region_filter_on else "Highest region", ACCENT),
            (False, "Other regions", GREY),
        ]:
            d = reg_rates[reg_rates["Region"].isin(focus) == is_focus]
            if d.empty:
                continue
            fig.add_trace(go.Bar(
                x=d[RATE], y=d["Pos"], orientation="h", name=name, marker_color=color,
                text=[f"<b>{t}</b>" if is_focus else t for t in d["Label"]],
                textposition="outside", cliponaxis=False,
                customdata=d[["Region"]],
                hovertemplate="%{customdata[0]}<br>%{x:.2f}%<extra></extra>",
            ))
        ref = wrate(reg_data)
        fig.add_vline(x=ref, line_dash="dash", line_color=RED,
                      annotation_text=f"All regions: {ref:.1f}%", annotation_position="top")
        fig.update_yaxes(
            tickmode="array", tickvals=reg_rates["Pos"].tolist(),
            ticktext=[f"<b>{r}</b>" if r in focus else r for r in reg_rates["Region"]],
            autorange="reversed", showgrid=False,
        )
        fig.update_xaxes(range=[0, reg_rates[RATE].max() * 1.2],
                         title="Weighted non-attendance (% of people in selection)")
        fig.update_layout(
            template=TEMPLATE, height=700, margin=dict(l=10, r=50, t=60, b=40), barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        show(fig)
        small(
            "Regions ranked from highest to lowest weighted share not attending school (2022 APIS). "
            "Honors the age and sex filters. The highest region is highlighted (or your selected "
            "region). A regional difference does not explain why it exists. Source: PSA APIS 2022."
        )

# --- 2b Age groups ---
age_data = subset(nat_df, regions=sel_regions, sexes=sel_sexes)
age_fig, age_rates = dot_plot(age_data, "Age Group", all_ages, sel_ages, 320, "All ages")
age_fig.update_yaxes(showgrid=False)
top_age_row = age_rates.loc[age_rates[RATE].idxmax()]
young = wrate(age_data[age_data["Age Group"].isin(YOUNGER_BANDS)])
old = wrate(age_data[age_data["Age Group"] == "19–24"])

with right:
    with card("age"):
        st.subheader(
            f"Non-attendance is highest among ages {top_age_row['Age Group']} "
            f"({top_age_row[RATE]:.1f}%)"
        )
        show(age_fig)
        st.markdown(
            '<div class="mini-row">'
            f'<div class="mini"><span>Ages 5–15 (3 bands combined)</span><b>{"n/a" if np.isnan(young) else f"{young:.1f}%"}</b></div>'
            f'<div class="mini"><span>Ages 19–24</span><b>{"n/a" if np.isnan(old) else f"{old:.1f}%"}</b></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        small(
            "Sorted highest to lowest. Honors the region and sex filters. Age groups are bands, not "
            "exact ages, so the dots are not a trend. Ages 19–24 are better read as a transition "
            "group: a high estimate does not automatically signal a school-access problem."
        )

    # --- 2c Sex: weighted rates (headline) + age distribution of non-attending people (violin) ---
    sex_data = subset(nat_df, regions=sel_regions, ages=sel_ages)
    sex_rates = rate_by(sex_data, "Sex").sort_values(RATE, ascending=False).reset_index(drop=True)
    na_ages = sex_data[sex_data["Attendance Status"] == "Not attending"].copy()
    na_ages["Age as of Last Birthday"] = pd.to_numeric(na_ages["Age as of Last Birthday"], errors="coerce")
    na_ages = na_ages.dropna(subset=["Age as of Last Birthday"])

    with card("sex"):
        if len(sex_rates) >= 2:
            st.subheader(
                f"{sex_rates.loc[0, 'Sex']}s had a higher non-attendance rate "
                f"({sex_rates.loc[0, RATE]:.1f}%) than {sex_rates.loc[1, 'Sex'].lower()}s "
                f"({sex_rates.loc[1, RATE]:.1f}%)"
            )
        else:
            st.subheader("Non-attendance by sex")

        # weighted rates stay visible as numbers; the violin adds the age picture
        st.markdown(
            '<div class="mini-row">'
            + "".join(
                f'<div class="mini"><span>{r["Sex"]}: weighted rate</span><b>{r[RATE]:.1f}%</b></div>'
                for _, r in sex_rates.iterrows()
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        for sex, color in [("Male", BLUE), ("Female", PINK)]:
            d = na_ages[na_ages["Sex"] == sex]
            if d.empty:
                continue
            fig.add_trace(go.Violin(
                x=[sex] * len(d), y=d["Age as of Last Birthday"], name=sex,
                fillcolor=color, line_color="#27303B", opacity=0.95 if sex in sel_sexes else 0.3,
                box_visible=True, meanline_visible=False, points=False, spanmode="hard",
                hovertemplate=f"{sex}<br>Age %{{y}}<extra></extra>",
            ))
        fig.update_layout(
            template=TEMPLATE, height=300, showlegend=False, margin=dict(l=45, r=20, t=20, b=40),
            yaxis_title="Age of non-attending people (years)", violingap=0.25,
        )
        show(fig)
        small(
            "The violin shows the <b>unweighted</b> age distribution of sampled people who were not "
            "attending (the box marks the median and quartiles; wider means more respondents at "
            "that age). It describes the sample, not a population rate: use the weighted rates above "
            "to compare males and females. Honors the region and age filters. A difference does not "
            "explain why it exists."
        )

# ------------------------------------------------------------------ #
# 3. INSIGHT
# ------------------------------------------------------------------ #
section("3", "Insight: what reasons were recorded?", "s3")
reason_df = with_reasons(sel_df)
sel_reasons = reason_shares(reason_df)

if sel_reasons.empty:
    note_box("No recorded reasons for this selection.", "info")
else:
    top10 = sel_reasons.head(10).copy()
    top10["Cumulative (%)"] = top10["Share (%)"].cumsum()
    top10["Short"] = top10["Why not attending school"].map(short)

    # reasons by sex (mirrored bar): honors region + age filters, both sexes shown
    mir_df = with_reasons(subset(nat_df, regions=sel_regions, ages=sel_ages))
    mir_top = reason_shares(mir_df).head(10)["Why not attending school"].tolist()
    mir = (
        mir_df[mir_df["Why not attending school"].isin(mir_top)]
        .groupby(["Why not attending school", "Sex"])[W].sum().reset_index()
    )
    denoms = mir_df.groupby("Sex")[W].sum()
    mir["Share"] = 100 * mir[W] / mir["Sex"].map(denoms)
    mir["Short"] = mir["Why not attending school"].map(short)
    order = [short(r) for r in mir_top]
    pivot = mir.pivot(index="Short", columns="Sex", values="Share").reindex(order)

    c_par, c_mir = st.columns(2)

    with c_par:
        with card("pareto"):
            if len(top10) >= 2:
                st.subheader(
                    f"“{top10.loc[0, 'Short']}” and “{top10.loc[1, 'Short']}” are the most frequently "
                    f"recorded reasons, together {top10.loc[1, 'Cumulative (%)']:.0f}% of recorded reasons"
                )
            else:
                st.subheader(f"“{top10.loc[0, 'Short']}” is the most frequently recorded reason")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=top10["Short"], y=top10["Share (%)"], name="Share of recorded reasons (%)",
                marker_color=ACCENT, text=top10["Share (%)"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=top10["Short"], y=top10["Cumulative (%)"], name="Cumulative share (%)",
                mode="lines+markers", line=dict(color=RED, width=3), marker=dict(size=8),
            ), secondary_y=True)
            fig.update_layout(
                template=TEMPLATE, height=520, margin=dict(l=45, r=45, t=60, b=120),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_xaxes(tickangle=-35)
            fig.update_yaxes(title_text="Share of recorded reasons (%)", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative share (%)", range=[0, 105], secondary_y=True,
                             showgrid=False)
            show(fig)
            small(
                "Honors all three filters. Denominator: weighted people not attending school who "
                "have a recorded reason. Bars show shares of reasons, not rates among all people. "
                "“Finished schooling” matters for interpreting the 19–24 group. Reasons are "
                "reported, not proven causes."
            )

    with c_mir:
        with card("mirror"):
            if {"Male", "Female"} <= set(pivot.columns):
                diff = pivot["Female"].fillna(0) - pivot["Male"].fillna(0)
                big = diff.abs().idxmax()
                st.subheader(
                    f"The mix of reasons differs most for “{big}”: "
                    f"{pivot.loc[big, 'Male']:.1f}% of males vs {pivot.loc[big, 'Female']:.1f}% of females"
                )
            else:
                st.subheader("Reasons recorded, by sex")

            fig = go.Figure()
            for sex, color, sign in [("Male", BLUE, -1), ("Female", PINK, 1)]:
                sub = mir[mir["Sex"] == sex]
                if sub.empty:
                    continue
                fig.add_trace(go.Bar(
                    y=sub["Short"], x=sub["Share"] * sign, orientation="h", name=sex,
                    marker_color=color, marker_line=dict(color="#FFFFFF", width=1),
                    text=sub["Share"].map(lambda v: f"{v:.1f}%"), textposition="outside",
                    customdata=sub[["Share"]],
                    hovertemplate=f"{sex}<br>%{{y}}<br>Share within {sex.lower()}s: %{{customdata[0]:.2f}}%<extra></extra>",
                ))
            mx = mir["Share"].max()
            fig.update_layout(
                barmode="overlay", template=TEMPLATE, height=520,
                xaxis_title="Share of non-attending people with a recorded reason (%)",
                yaxis=dict(categoryorder="array", categoryarray=order, autorange="reversed",
                           showgrid=False),
                margin=dict(l=40, r=40, t=60, b=65),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.update_xaxes(
                tickmode="array", tickvals=[-mx, -mx / 2, 0, mx / 2, mx],
                ticktext=[f"{mx:.0f}%", f"{mx / 2:.0f}%", "0%", f"{mx / 2:.0f}%", f"{mx:.0f}%"],
                range=[-mx * 1.3, mx * 1.3], zeroline=True, zerolinewidth=2, zerolinecolor="#666",
            )
            show(fig)
            small(
                "Honors the region and age filters (both sexes always shown). Males on the left, "
                "females on the right. Each bar is the share of that sex's non-attending people who "
                "gave the reason: it compares the <i>mix</i> of reasons within each sex, not how "
                "many males or females were affected."
            )

# ------------------------------------------------------------------ #
# 4. IMPLICATION (national, unfiltered evidence)
# ------------------------------------------------------------------ #
section("4", "Implication: why this matters for DepEd", "s4")
top_nat_reason = short(nat_reasons.loc[0, "Why not attending school"])

claim_boundary = pd.DataFrame([
    {
        "Observed pattern": f"Overall weighted non-attendance is {nat_rate:.2f}%.",
        "What the data supports": "A weighted descriptive estimate for the population represented by the 2022 APIS.",
        "What it does not prove": "The current 2026 rate, a causal explanation, or an official SDG benchmark.",
    },
    {
        "Observed pattern": f"{nat_hi['Region']} has the highest regional estimate ({nat_hi[RATE]:.2f}%).",
        "What the data supports": "A comparison of weighted estimates across regions in this dataset.",
        "What it does not prove": "Why the region is higher, or which intervention will work.",
    },
    {
        "Observed pattern": f"Ages {nat_top_age} have a much higher estimate than younger groups.",
        "What the data supports": "An age-group difference in current attendance status.",
        "What it does not prove": "That everyone in this group dropped out or has not completed education.",
    },
    {
        "Observed pattern": "Respondents report a range of reasons for not attending.",
        "What the data supports": "Recorded reasons and their weighted shares.",
        "What it does not prove": "That a reason is the sole or causal explanation.",
    },
    {
        "Observed pattern": "Regional and demographic estimates differ.",
        "What the data supports": "Descriptive comparisons between groups.",
        "What it does not prove": "Statistical significance; design-adjusted uncertainty is not calculated.",
    },
])

c4a, c4b = st.columns([1, 1.25])
with c4a:
    with card("implication"):
        st.markdown('<div class="eyebrow">National picture (unfiltered)</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
- **Scale.** Roughly 1 in 5 people aged 5–24 ({nat_rate:.1f}%) were not attending school in 2022, so non-attendance is not a marginal issue in the survey population.
- **Age matters.** The estimate is highest for ages {nat_top_age} ({nat_ages[nat_top_age]:.1f}%), where “{top_nat_reason}” is the most frequently recorded reason nationally ({nat_reasons.loc[0, 'Share (%)']:.1f}%). This group should be read as a transition group, with compulsory-school-age children kept central to SDG 4.1.
- **Place matters.** Regional estimates span about {nat_hi[RATE] - nat_lo[RATE]:.1f} percentage points ({nat_lo['Region']} to {nat_hi['Region']}), which suggests where closer review may be worthwhile, but not why the gaps exist.
- **Reasons are mixed.** No single recorded reason accounts for most non-attendance, so a single program is unlikely to address all groups.
            """
        )
with c4b:
    with card("claims"):
        st.markdown("##### What the evidence supports, and what it does not prove")
        st.dataframe(claim_boundary, hide_index=True)

# ------------------------------------------------------------------ #
# 5. ACTION / NEXT STEP
# ------------------------------------------------------------------ #
section("5", "Action: what should be considered next?", "s5")
a1, a2 = st.columns(2)
with a1:
    with card("action_ok"):
        st.markdown("##### Directly supported by this evidence")
        st.markdown(
            f"""
1. **Use age-appropriate comparisons.** Keep compulsory-school-age children central to the SDG 4.1 discussion and interpret ages 19–24 separately.
2. **Prioritize regions for closer review.** Treat regions with the highest estimates (e.g., {nat_hi['Region']}) as a *monitoring priority*, not as proof that an intervention is needed.
            """
        )
with a2:
    with card("action_more"):
        st.markdown("##### Needs additional evidence first")
        st.markdown(
            """
3. **Investigate regional patterns** with local school participation, distance/access, household finance, and education-service data before choosing interventions.
4. **Examine reasons by age group and region** (e.g., whether financial concerns differ) before proposing targeted support.
5. **Compare with newer data** only when definitions and populations are comparable to the 2022 APIS.
            """
        )

# ------------------------------------------------------------------ #
# Limitations, definitions, sources
# ------------------------------------------------------------------ #
section("•", "Limitations, definitions and sources", "s6")

limits = pd.DataFrame([
    {"Issue / limitation": "Data are from 2022",
     "Why it matters": "Results may not describe the situation in 2026.",
     "How the dashboard handles it": "2022 is stated in the titles, warning note and footer."},
    {"Issue / limitation": "Non-attendance is not dropout",
     "Why it matters": "Some people aged 19–24 have completed their education.",
     "How the dashboard handles it": "Age bands are shown separately; “Finished schooling” is shown as a recorded reason."},
    {"Issue / limitation": "Weighted descriptive estimates only",
     "Why it matters": "No design-adjusted standard errors, confidence intervals or significance tests.",
     "How the dashboard handles it": "Differences are described, not called significant; small-sample warning appears when filters leave few records."},
    {"Issue / limitation": "Reasons are self-reported",
     "Why it matters": "A recorded reason may not be the only or the causal factor.",
     "How the dashboard handles it": "Language avoids causal claims; denominators are stated under each chart."},
    {"Issue / limitation": "Aggregated, anonymized data",
     "Why it matters": "Individuals must not be identifiable.",
     "How the dashboard handles it": "Only grouped percentages and aggregate counts are displayed."},
])
with card("limits"):
    st.dataframe(limits, hide_index=True)
    with st.expander("Metric definitions and data notes"):
        st.markdown(
            f"""
- **Weighted non-attendance rate (%):** sum of `{W}` for people marked *Not attending* ÷ sum of `{W}` for all people in the group × 100.
- **Share of recorded reasons (%):** weighted people not attending with a recorded reason, by reason ÷ all weighted people not attending with a recorded reason × 100. The mirrored chart uses each sex's own total as the denominator.
- **Gap between regions:** highest regional weighted non-attendance rate minus the lowest, in percentage points (a descriptive spread, not a significance test).
- **Ages 5–15:** the 5–9, 10–12 and 13–15 bands combined.
- **Violin plot:** unweighted ages of sampled people who were not attending; shows the shape of the sample, not a weighted population distribution.
- **Records:** {len(raw):,} in the cleaned file; {len(nat_df):,} retained (valid attendance status and positive weight).
            """
        )
    st.markdown(
        "**Source:** Philippine Statistics Authority (2023). *Four out of five children aged 5 to 24 "
        f"years were attending school for school year 2022 to 2023.* [{PSA_URL}]({PSA_URL}). "
        "Data: 2022 Annual Poverty Indicators Survey (APIS) public-use file, cleaned for this project. "
        "SDG reference: UN DESA, *Goal 4*, https://sdgs.un.org/goals/goal4"
    )
    small(
        "Weighted descriptive estimates for 2022; not current 2026 figures. Patterns and reported "
        "reasons do not prove cause and effect."
    )
