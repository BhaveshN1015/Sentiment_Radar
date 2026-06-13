"""
streamlit_app.py — Sentiment Radar
Clean dark-theme dashboard: Interface.py polish + Dashboard.py clarity.
Run: streamlit run streamlit_app.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Auto-load .env from project root ─────────────────────────
def _load_env_app():
    from pathlib import Path
    import os
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip(); val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_env_app()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Sentiment Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load heavy modules once ───────────────────────────────────
@st.cache_resource(show_spinner="⏳  Loading model — this takes ~30 seconds on first run…")
def _load_backend():
    from app.inference import analyze_platforms
    from app.scraper   import scrape_all
    return analyze_platforms, scrape_all

# ── Palette ───────────────────────────────────────────────────
C = dict(
    pos="#34d399", neg="#f87171", neu="#fbbf24", unc="#9ca3af",
    plat=["#a78bfa","#38bdf8","#fb923c","#4ade80"],
    bg="#0a0a0f", card="#13131f", border="#1e1e30", muted="#4b5563",
    text="#e8e8f0", accent="#a78bfa",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;background:{C['bg']};color:{C['text']}}}
.stApp{{background:{C['bg']}}}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{{background:{C['card']};border-radius:12px;padding:4px;gap:4px;border:1px solid {C['border']}}}
.stTabs [data-baseweb="tab"]{{background:transparent;border-radius:8px;color:{C['muted']};font-size:.82rem;font-weight:500;padding:.4rem 1rem}}
.stTabs [aria-selected="true"]{{background:{C['border']} !important;color:{C['text']} !important}}
/* Button */
.stButton>button{{background:linear-gradient(135deg,#7c3aed,#2563eb);color:#fff;border:none;border-radius:10px;
  padding:.55rem 2rem;font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;width:100%;transition:opacity .2s}}
.stButton>button:hover{{opacity:.88}}
/* Input */
.stTextInput>div>div>input{{background:{C['card']} !important;border:1px solid #2a2a3e !important;border-radius:10px !important;
  color:{C['text']} !important;font-size:1rem !important;padding:.6rem 1rem !important}}
.stTextInput>div>div>input:focus{{border-color:#7c3aed !important;box-shadow:0 0 0 2px rgba(124,58,237,.25) !important}}
/* Selectbox */
.stSelectbox>div>div{{background:{C['card']} !important;border-color:#2a2a3e !important}}
/* Dataframe */
.stDataFrame{{border:1px solid {C['border']};border-radius:10px}}
div[data-testid="stMetricValue"]{{font-family:'Syne',sans-serif}}
/* Sidebar */
section[data-testid="stSidebar"]{{background:#0d0d18;border-right:1px solid {C['border']}}}
/* Custom classes */
.hero-title{{font-family:'Syne',sans-serif;font-weight:800;font-size:2.6rem;
  background:linear-gradient(135deg,#a78bfa,#38bdf8,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;margin-bottom:.2rem}}
.section-lbl{{font-family:'Syne',sans-serif;font-size:.7rem;font-weight:700;color:{C['accent']};
  text-transform:uppercase;letter-spacing:.12em;margin-bottom:.75rem}}
.kpi-card{{background:{C['card']};border:1px solid {C['border']};border-radius:14px;padding:1.2rem 1.4rem;position:relative;overflow:hidden}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#a78bfa,#38bdf8)}}
.kpi-lbl{{font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:{C['muted']};margin-bottom:.3rem}}
.kpi-val{{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;line-height:1;margin-bottom:.2rem}}
.kpi-sub{{font-size:.7rem;color:{C['muted']}}}
.ccard{{background:{C['card']};border:1px solid {C['border']};border-left:3px solid transparent;
  border-radius:10px;padding:.85rem 1rem;margin-bottom:.45rem;font-size:.83rem;line-height:1.6;color:#c4c4d4}}
.ccard.positive{{border-left-color:{C['pos']}}}
.ccard.negative{{border-left-color:{C['neg']}}}
.ccard.neutral{{border-left-color:{C['neu']}}}
.ccard.uncertain{{border-left-color:{C['unc']}}}
.cmeta{{font-size:.66rem;color:{C['muted']};margin-top:.35rem;display:flex;gap:.6rem}}
.pill{{display:inline-block;font-size:.6rem;font-weight:600;padding:.1rem .4rem;border-radius:20px;background:#1e1e30}}
.chip{{display:inline-block;background:{C['card']};border:1px solid #2a2a3e;border-radius:20px;
  padding:.2rem .75rem;font-size:.76rem;color:{C['muted']};margin:.15rem;cursor:default}}
.gap-row{{background:{C['card']};border:1px solid {C['border']};border-radius:12px;padding:.9rem 1.1rem;margin-bottom:.45rem}}
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ─────────────────────────────────────────────
_LAY = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,20,.6)",
            font=dict(family="DM Sans", color=C['text'], size=12),
            margin=dict(t=18,b=18,l=18,r=18))

def _sent_color(s): return C.get(s[:3], C['unc'])

def donut(counts: dict, title="", h=270):
    labels = [k.capitalize() for k in counts]
    colors = [_sent_color(k) for k in counts]
    fig = go.Figure(go.Pie(
        labels=labels, values=list(counts.values()), hole=.55,
        marker=dict(colors=colors, line=dict(color=C['bg'], width=2)),
        textinfo="percent", textfont=dict(size=11, color=C['text']),
        hovertemplate="<b>%{label}</b><br>%{value} comments<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_LAY, height=h, showlegend=False)
    if title: fig.update_layout(title=dict(text=title, font=dict(size=11,color=C['muted']),x=.5))
    return fig

def grouped_bar(df: pd.DataFrame, h=290):
    platforms = df["platform"].unique()
    fig = go.Figure()
    for sent, color in [("positive",C['pos']),("neutral",C['neu']),("negative",C['neg'])]:
        fig.add_trace(go.Bar(
            name=sent.capitalize(),
            x=list(platforms),
            y=[len(df[(df.platform==p)&(df.sentiment==sent)]) for p in platforms],
            marker_color=color, marker_line=dict(width=0),
        ))
    fig.update_layout(**_LAY, height=h, barmode="group", showlegend=True,
        legend=dict(orientation="h",y=1.1,x=.5,xanchor="center",font=dict(size=11,color=C['text'])),
        xaxis=dict(gridcolor=C['border']), yaxis=dict(gridcolor=C['border']))
    return fig

def conf_hist(df: pd.DataFrame, h=265):
    fig = go.Figure()
    for sent, color in [("positive",C['pos']),("neutral",C['neu']),("negative",C['neg'])]:
        sub = df[df.sentiment==sent]["confidence"]
        if not sub.empty:
            fig.add_trace(go.Histogram(x=sub, name=sent.capitalize(),
                nbinsx=15, marker_color=color, opacity=.75))
    fig.update_layout(**_LAY, height=h, barmode="overlay", showlegend=True,
        legend=dict(orientation="h",y=1.1,x=.5,xanchor="center",font=dict(size=11,color=C['text'])),
        xaxis=dict(title="Confidence %", gridcolor=C['border']),
        yaxis=dict(title="Count", gridcolor=C['border']))
    return fig

def radar_chart(df: pd.DataFrame, h=290):
    platforms = df["platform"].unique().tolist()
    cats = ["Positive %","Neutral %","Negative %","Avg Confidence"]
    fig  = go.Figure()
    for i, p in enumerate(platforms):
        sub   = df[df.platform==p]; total = max(len(sub),1)
        vals  = [
            round(len(sub[sub.sentiment=="positive"])/total*100,1),
            round(len(sub[sub.sentiment=="neutral"]) /total*100,1),
            round(len(sub[sub.sentiment=="negative"])/total*100,1),
            round(sub["confidence"].mean(),1),
        ]
        clr = C['plat'][i % len(C['plat'])]
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            name=p, fill="toself", opacity=.55,
            line=dict(color=clr, width=1.5),
        ))
    fig.update_layout(**_LAY, height=h, showlegend=True,
        legend=dict(orientation="h",y=-0.15,x=.5,xanchor="center",font=dict(size=11,color=C['text'])),
        polar=dict(
            bgcolor=C['card'],
            radialaxis=dict(visible=True, gridcolor=C['border'], color=C['muted'], range=[0,100]),
            angularaxis=dict(gridcolor=C['border'], color="#9ca3af"),
        ))
    return fig

def gap_bar(df: pd.DataFrame):
    rows = []
    for p in df["platform"].unique():
        sub = df[df.platform==p]; total = max(len(sub),1)
        score = round(
            (len(sub[sub.sentiment=="positive"]) - len(sub[sub.sentiment=="negative"])) / total * 100, 1
        )
        rows.append({"Platform": p, "Score": score})
    sdf    = pd.DataFrame(rows).sort_values("Score", ascending=True)
    colors = [C['pos'] if s >= 0 else C['neg'] for s in sdf["Score"]]
    fig = go.Figure(go.Bar(
        x=sdf["Score"], y=sdf["Platform"], orientation="h",
        marker_color=colors, marker_line=dict(width=0),
        text=[f"{s:+.1f}%" for s in sdf["Score"]], textposition="outside",
        textfont=dict(size=11, color=C['text']),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=C['border'], line_width=1)
    fig.update_layout(**_LAY, height=max(200, len(rows)*70),
        xaxis=dict(title="Positivity score (positive% − negative%)",
                   gridcolor=C['border'], ticksuffix="%"),
        yaxis=dict(gridcolor=C['border']))
    return fig

# ── Data helpers ──────────────────────────────────────────────
def build_df(results: dict) -> pd.DataFrame:
    rows = []
    for platform, entries in results.items():
        for e in entries:
            rows.append(dict(platform=platform, comment=e["comment"],
                             sentiment=e["sentiment"], confidence=e["confidence"]))
    return pd.DataFrame(rows)

def counts(df, platform=None):
    sub  = df if platform is None else df[df.platform==platform]
    base = dict(positive=0, neutral=0, negative=0, uncertain=0)
    for s, n in sub["sentiment"].value_counts().items():
        base[s] = int(n)
    return base

def pos_score(c: dict) -> float:
    t = sum(c.values()) or 1
    return round((c.get("positive",0) - c.get("negative",0)) / t * 100, 1)

def dominant(c: dict):
    main = {k: v for k, v in c.items() if k != "uncertain"}
    if not main or max(main.values()) == 0: return "uncertain", 0
    d = max(main, key=main.get)
    return d, round(main[d] / (sum(c.values()) or 1) * 100, 1)

def comment_cards(rows, max_show=10):
    for comment, sentiment, confidence in rows[:max_show]:
        sc = _sent_color(sentiment)
        st.markdown(f"""
        <div class="ccard {sentiment}">
            {comment}
            <div class="cmeta">
                <span style="color:{sc};font-weight:600">{sentiment.capitalize()}</span>
                <span class="pill">{confidence:.1f}% confidence</span>
            </div>
        </div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
def sidebar_settings():
    """
    Sidebar controls. YouTube activates automatically when YOUTUBE_API_KEY
    is set in .env — the key is never shown or entered in the UI.
    """
    import os
    yt_configured = bool(os.getenv("YOUTUBE_API_KEY", "").strip())
    with st.sidebar:
        st.markdown(f'<div class="hero-title" style="font-size:1.4rem">⚙️ Settings</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div class="section-lbl">Platforms</div>', unsafe_allow_html=True)

        # YouTube — auto-enabled if key in .env, no UI key input ever shown
        if yt_configured:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.5rem;padding:.4rem 0">
                <span style="color:#34d399;font-size:.75rem">●</span>
                <span style="font-size:.82rem;color:{C['text']}">YouTube</span>
                <span style="font-size:.65rem;color:{C['muted']};margin-left:auto">active</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="font-size:.72rem;color:{C['muted']};padding:.3rem 0;
                        border:1px dashed #2a2a3e;border-radius:8px;padding:.5rem .7rem;
                        margin-bottom:.3rem">
                📺 <b>YouTube</b> — add <code>YOUTUBE_API_KEY</code> to your
                <code>.env</code> file to enable
            </div>""", unsafe_allow_html=True)

        use_tw = st.toggle("Bluesky", value=False,
                           help="Bluesky public posts — free, no key needed")

        st.markdown("---")
        st.markdown('<div class="section-lbl">Comments per platform</div>', unsafe_allow_html=True)
        n_comments = st.slider("Comments per platform", 20, 100, 50, step=10, label_visibility="collapsed")

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:.7rem;color:{C['muted']};line-height:1.6">
            <b style="color:{C['accent']}">Sentiment Radar</b><br>
            BiLSTM + Self-Attention model<br>
            Trained on 6 social platforms<br>
            Classes: Positive · Neutral · Negative<br>
            Confidence threshold: 50%
        </div>""", unsafe_allow_html=True)

    return use_tw, n_comments

# ── Main ──────────────────────────────────────────────────────
def main():
    use_tw, n_comments = sidebar_settings()

    # Hero
    st.markdown("""
    <div style="padding:1.2rem 0 .5rem">
        <div class="hero-title">📡 Sentiment Radar</div>
        <div style="font-size:.9rem;color:#4b5563;margin-bottom:1.2rem">
            Real-time sentiment analysis · Reddit · HackerNews · Dev.to · YouTube · Bluesky
            <span id="yt-badge"></span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Search row
    ci, cb = st.columns([4, 1])
    with ci:
        topic = st.text_input("topic", label_visibility="collapsed",
                               placeholder="Enter any topic — e.g. Artificial Intelligence, Bitcoin…",
                               key="topic")
    with cb:
        go_btn = st.button("Analyze →")

    # Example chips
    examples = ["Artificial Intelligence","Bitcoin","Climate Change",
                "Electric Vehicles","Remote Work","ChatGPT"]
    st.markdown("".join(f'<span class="chip">{e}</span>' for e in examples),
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Empty state
    if not (go_btn and topic.strip()):
        st.markdown(f"""
        <div style="text-align:center;padding:4rem 0;color:{C['muted']}">
            <div style="font-size:3rem;margin-bottom:.8rem">🔍</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.05rem;color:#6b7280">
                Enter a topic and click <b style="color:{C['accent']}">Analyze →</b>
            </div>
            <div style="font-size:.78rem;color:#374151;margin-top:.4rem">
                Fetches ~{n_comments} comments per platform · no API key needed for Reddit, HN, Dev.to
            </div>
        </div>""", unsafe_allow_html=True)
        return

    topic = topic.strip()

    # Load backend
    analyze_platforms, scrape_all = _load_backend()

    # Scrape
    with st.spinner(f"Scraping comments for **{topic}** …"):
        raw = scrape_all(
            topic,
            comments_per_platform=n_comments,
            include_twitter=use_tw,
        )

    total_scraped = sum(len(v) for v in raw.values())
    if total_scraped == 0:
        st.error("No comments found. Try a broader topic or check your internet connection.")
        return

    # Platform fetch counts info box
    fetch_info = "  ·  ".join(
        f"**{p}** {len(v)} comments" for p, v in raw.items()
    )
    st.info(f"📥  Fetched: {fetch_info}", icon=None)

    # Inference
    with st.spinner("Running sentiment analysis…"):
        results = analyze_platforms(raw)

    df = build_df(results)
    if df.empty:
        st.error("Analysis returned no results. Please try again.")
        return

    platforms   = df["platform"].unique().tolist()
    ov          = counts(df)
    total       = len(df)
    dom, dom_pct = dominant(ov)
    avg_conf    = round(df["confidence"].mean(), 1)
    ps          = pos_score(ov)

    # ── KPI row ───────────────────────────────────────────────
    st.markdown('<div class="section-lbl">Overview</div>', unsafe_allow_html=True)
    kpi_cols = st.columns(5)
    kpis = [
        ("Total Comments",   str(total),              f"{len(platforms)} platforms",   C['accent']),
        ("Dominant Mood",    dom.capitalize(),         f"{dom_pct}% of comments",       _sent_color(dom)),
        ("Avg Confidence",   f"{avg_conf}%",           "model certainty",               "#38bdf8"),
        ("Positivity Score", f"{ps:+.1f}%",            "positive minus negative",       C['pos'] if ps>=0 else C['neg']),
        ("Platforms",        str(len(platforms)),      " · ".join(platforms),           "#fb923c"),
    ]
    for col, (lbl, val, sub, color) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">{lbl}</div>
                <div class="kpi-val" style="color:{color}">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab_labels = ["🌐 Overall","📊 By Platform","💬 Comments","⚖️ Sentiment Gap","🎯 Deep Dive"]

    # ── Persist active tab across reruns via JS injection ─────
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = 0

    def _set_tab(idx: int):
        st.session_state["active_tab"] = idx

    tabs = st.tabs(tab_labels)

    # Inject JS to auto-click the correct tab button after every rerun.
    # Streamlit renders tabs as buttons with data-baseweb="tab".
    _active = st.session_state["active_tab"]
    st.markdown(f"""
    <script>
    (function() {{
        function clickTab() {{
            const tabList = window.parent.document.querySelectorAll(
                '[data-baseweb="tab"]');
            if (tabList.length > {_active}) {{
                tabList[{_active}].click();
            }}
        }}
        // Run after Streamlit finishes rendering
        if (document.readyState === "complete") {{
            setTimeout(clickTab, 120);
        }} else {{
            window.addEventListener("load", function() {{
                setTimeout(clickTab, 120);
            }});
        }}
    }})();
    </script>
    """, unsafe_allow_html=True)

    # ── Tab 1: Overall ────────────────────────────────────────
    with tabs[0]:
        if st.session_state.get("active_tab") != 0:
            _set_tab(0)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.markdown('<div class="section-lbl">Overall sentiment distribution</div>', unsafe_allow_html=True)
            disp = {k: v for k, v in ov.items() if v > 0}
            st.plotly_chart(donut(disp, f"All platforms · {total} comments"),
                            width="stretch", config={"displayModeBar": False})
            for sent in ["positive","neutral","negative"]:
                n   = ov.get(sent, 0)
                pct = round(n / total * 100, 1) if total else 0
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:.45rem .8rem;background:{C['card']};border-radius:8px;
                            margin-bottom:.3rem;border:1px solid {C['border']}">
                    <span style="color:{_sent_color(sent)};font-weight:600;font-size:.82rem">{sent.capitalize()}</span>
                    <span style="font-family:'Syne',sans-serif;color:{C['text']}">{n}</span>
                    <span style="color:{C['muted']};font-size:.76rem">{pct}%</span>
                </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-lbl">Platform breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(grouped_bar(df), width="stretch", config={"displayModeBar": False})
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-lbl">Platform radar</div>', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(df), width="stretch", config={"displayModeBar": False})

    # ── Tab 2: By Platform ────────────────────────────────────
    with tabs[1]:
        if st.session_state.get("active_tab") != 1:
            _set_tab(1)
        st.markdown("<br>", unsafe_allow_html=True)
        for i in range(0, len(platforms), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i+j >= len(platforms): break
                p  = platforms[i+j]
                c  = counts(df, p)
                t  = sum(c.values())
                ac = round(df[df.platform==p]["confidence"].mean(), 1)
                clr= C['plat'][(i+j) % len(C['plat'])]
                with col:
                    st.markdown(f"""
                    <div style="font-family:'Syne',sans-serif;font-size:.76rem;text-transform:uppercase;
                                letter-spacing:.1em;color:{clr};margin-bottom:.3rem">
                        {p} <span style="color:{C['muted']};font-size:.63rem;text-transform:none;
                        letter-spacing:0">· {t} comments · avg {ac}% confidence</span>
                    </div>""", unsafe_allow_html=True)
                    st.plotly_chart(donut({k:v for k,v in c.items() if v>0}, h=230),
                                   width="stretch", config={"displayModeBar": False})
                    m1, m2, m3 = st.columns(3)
                    for mc, sent in zip([m1,m2,m3], ["positive","neutral","negative"]):
                        pct = round(c[sent]/(t or 1)*100,1)
                        with mc:
                            st.markdown(f"""
                            <div style="text-align:center;padding:.35rem;background:{C['card']};
                                        border-radius:8px;border:1px solid {C['border']}">
                                <div style="color:{_sent_color(sent)};font-size:.6rem;font-weight:600">{sent[:3].upper()}</div>
                                <div style="font-family:'Syne',sans-serif;font-size:.95rem;color:{C['text']}">{c[sent]}</div>
                                <div style="font-size:.58rem;color:{C['muted']}">{pct}%</div>
                            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Tab 3: Comments ───────────────────────────────────────
    with tabs[2]:
        if st.session_state.get("active_tab") != 2:
            _set_tab(2)
        st.markdown("<br>", unsafe_allow_html=True)

        # Use session_state keys so selectbox changes do NOT trigger
        # a full page re-render that scrolls back to the top.
        if "cmts_platform"  not in st.session_state: st.session_state["cmts_platform"]  = "All"
        if "cmts_sentiment" not in st.session_state: st.session_state["cmts_sentiment"] = "All"
        if "cmts_show"      not in st.session_state: st.session_state["cmts_show"]      = 10

        f1, f2, f3 = st.columns([1.5, 1.5, 1])
        with f1:
            sel_p = st.selectbox(
                "Platform", ["All"] + platforms,
                index=(["All"] + platforms).index(st.session_state["cmts_platform"])
                      if st.session_state["cmts_platform"] in (["All"] + platforms) else 0,
                key="cmts_platform",
                on_change=lambda: _set_tab(2),
            )
        with f2:
            sent_opts = ["All", "Positive", "Negative", "Neutral", "Uncertain"]
            sel_s = st.selectbox(
                "Sentiment", sent_opts,
                index=sent_opts.index(st.session_state["cmts_sentiment"])
                      if st.session_state["cmts_sentiment"] in sent_opts else 0,
                key="cmts_sentiment",
                on_change=lambda: _set_tab(2),
            )
        with f3:
            show_opts = [10, 25, 50]
            max_s = st.selectbox(
                "Show", show_opts,
                index=show_opts.index(st.session_state["cmts_show"])
                      if st.session_state["cmts_show"] in show_opts else 0,
                key="cmts_show",
                on_change=lambda: _set_tab(2),
            )

        fdf = df.copy()
        if sel_p != "All": fdf = fdf[fdf.platform  == sel_p]
        if sel_s != "All": fdf = fdf[fdf.sentiment == sel_s.lower()]

        st.markdown(
            f'<div style="font-size:.74rem;color:{C["muted"]};margin-bottom:1rem">' +
            f'Showing {min(max_s, len(fdf))} of {len(fdf)} comments</div>',
            unsafe_allow_html=True,
        )
        if fdf.empty:
            st.info("No comments match the selected filters.")
        else:
            comment_cards(fdf[["comment","sentiment","confidence"]].values.tolist(), max_s)

        with st.expander("📥 Download results as CSV"):
            st.download_button("Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"sentiment_{topic.replace(' ','_')}.csv",
                mime="text/csv")

    # ── Tab 4: Sentiment Gap ──────────────────────────────────
    with tabs[3]:
        if st.session_state.get("active_tab") != 3:
            _set_tab(3)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Positivity score by platform</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:.76rem;color:{C["muted"]};margin-bottom:1rem">'
                    'Positivity score = positive% − negative%. Higher = more positive community.</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(gap_bar(df), width="stretch", config={"displayModeBar": False})
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Per-platform stacked bars</div>', unsafe_allow_html=True)
        for p in sorted(platforms, key=lambda p: pos_score(counts(df,p)), reverse=True):
            c  = counts(df, p); t = sum(c.values()) or 1
            ps_p   = pos_score(c)
            ac_p   = round(df[df.platform==p]["confidence"].mean(),1)
            pw = round(c["positive"]/t*100,1)
            nw = round(c["neutral"] /t*100,1)
            gw = round(c["negative"]/t*100,1)
            sc = C['pos'] if ps_p >= 0 else C['neg']
            st.markdown(f"""
            <div class="gap-row">
                <div style="display:flex;justify-content:space-between;font-size:.77rem;margin-bottom:.55rem">
                    <span style="font-family:'Syne',sans-serif;color:{C['text']}">{p}</span>
                    <span>
                        <span style="color:{sc};font-weight:700">{ps_p:+.1f}%</span>
                        <span style="color:{C['muted']};font-size:.66rem;margin-left:.4rem">avg conf {ac_p}%</span>
                    </span>
                </div>
                <div style="display:flex;height:9px;border-radius:20px;overflow:hidden;gap:2px">
                    <div style="width:{pw}%;background:{C['pos']};border-radius:20px 0 0 20px"></div>
                    <div style="width:{nw}%;background:{C['neu']}"></div>
                    <div style="width:{gw}%;background:{C['neg']};border-radius:0 20px 20px 0"></div>
                </div>
                <div style="display:flex;gap:.8rem;margin-top:.45rem;font-size:.66rem">
                    <span style="color:{C['pos']}">● {pw}% positive</span>
                    <span style="color:{C['neu']}">● {nw}% neutral</span>
                    <span style="color:{C['neg']}">● {gw}% negative</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # Cross-platform summary KPIs
        st.markdown("<br>", unsafe_allow_html=True)
        scores   = {p: pos_score(counts(df, p)) for p in platforms}
        most_pos = max(scores, key=scores.get)
        least_pos= min(scores, key=scores.get)
        g1,g2,g3 = st.columns(3)
        for gcol, lbl, val, sub_txt, color in [
            (g1,"Most positive", most_pos,  f"{scores[most_pos]:+.1f}% score",  C['pos']),
            (g2,"Most negative", least_pos, f"{scores[least_pos]:+.1f}% score", C['neg']),
            (g3,"Gap",           f"{round(scores[most_pos]-scores[least_pos],1)}%",
             f"{most_pos} vs {least_pos}", "#38bdf8"),
        ]:
            with gcol:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-lbl">{lbl}</div>
                    <div class="kpi-val" style="color:{color};font-size:1.4rem">{val}</div>
                    <div class="kpi-sub">{sub_txt}</div>
                </div>""", unsafe_allow_html=True)

    # ── Tab 5: Deep Dive ──────────────────────────────────────
    with tabs[4]:
        if st.session_state.get("active_tab") != 4:
            _set_tab(4)
        st.markdown("<br>", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.markdown('<div class="section-lbl">Confidence distribution by sentiment</div>', unsafe_allow_html=True)
            st.plotly_chart(conf_hist(df), width="stretch", config={"displayModeBar": False})
        with d2:
            st.markdown('<div class="section-lbl">Volume vs positivity (bubble size = avg confidence)</div>', unsafe_allow_html=True)
            fig_b = go.Figure()
            for i, p in enumerate(platforms):
                c  = counts(df, p)
                ps_p = pos_score(c)
                ac_p = round(df[df.platform==p]["confidence"].mean(), 1)
                vc   = len(df[df.platform==p])
                clr  = C['plat'][i % len(C['plat'])]
                fig_b.add_trace(go.Scatter(
                    x=[vc], y=[ps_p], mode="markers+text",
                    marker=dict(size=ac_p/2, color=clr, opacity=.8,
                                line=dict(width=1, color=clr)),
                    text=[p], textposition="top center",
                    textfont=dict(size=10, color=C['text']),
                    name=p,
                    hovertemplate=f"<b>{p}</b><br>Volume: {vc}<br>Score: {ps_p:+.1f}%<br>Avg Conf: {ac_p}%<extra></extra>",
                ))
            fig_b.add_hline(y=0, line_color=C['border'], line_width=1, line_dash="dot")
            fig_b.update_layout(**_LAY, height=265,
                xaxis=dict(title="Comment volume", gridcolor=C['border']),
                yaxis=dict(title="Positivity score %", gridcolor=C['border'], ticksuffix="%"))
            st.plotly_chart(fig_b, width="stretch", config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">Full platform summary table</div>', unsafe_allow_html=True)
        summary = []
        for p in platforms:
            c  = counts(df, p); t = sum(c.values())
            ds, dp = dominant(c)
            summary.append({
                "Platform": p, "Total": t,
                "Positive": c["positive"], "Neutral": c["neutral"],
                "Negative": c["negative"], "Uncertain": c["uncertain"],
                "Dominant": ds.capitalize(),
                "Positivity score": f"{pos_score(c):+.1f}%",
                "Avg confidence": f"{round(df[df.platform==p]['confidence'].mean(),1)}%",
            })
        st.dataframe(pd.DataFrame(summary).set_index("Platform"), width="stretch")

    # Footer
    st.markdown(f"<br><hr style='border-color:{C['border']}'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;font-size:.68rem;color:#374151;padding:.4rem 0">
        Sentiment Radar · Topic: <b style="color:{C['accent']}">{topic}</b> ·
        {total} comments · {len(platforms)} platforms
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
