
import base64, json, pickle, time, uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import database as db
from recommender import CBFRecommender

@st.cache_data(ttl=60)
def _mongo_ok() -> bool:
    return db.is_connected()

@st.cache_data(ttl=3600)
def _load_valid_pids() -> frozenset:
    with open(str(ITEM_IDS), "rb") as f:
        return frozenset(pickle.load(f))

@st.cache_data(ttl=300)
def _get_popular_games(n: int = 60) -> list[dict]:
    if _mongo_ok():
        valid = _load_valid_pids()
        docs  = db.get_popular_items(n + 30)
        return [d for d in docs if d.get("product_id") in valid][:n]
    return []

@st.cache_data(ttl=60)
def _search_games(query: str, n: int = 30) -> list[dict]:
    if not query.strip():
        return []
    if _mongo_ok():
        valid = _load_valid_pids()
        docs  = db.search_items(query, n + 20)
        return [d for d in docs if d.get("product_id") in valid][:n]
    return []

@st.cache_data(ttl=60)
def _get_db_stats() -> tuple[dict, int, int]:
    if not _mongo_ok():
        return {}, 0, 0
    shard_info, total_items, total_uat = {}, 0, 0
    try:
        shard_info = db.get_shard_info()
    except Exception:
        pass
    try:
        client = db.get_client()
        d = client[db.DB_NAME]
        total_items = d[db.ITEMS_COLLECTION].count_documents({})
        total_uat   = d[db.UAT_COLLECTION].count_documents({})
    except Exception:
        pass
    return shard_info, total_items, total_uat

DATA_DIR     = Path(__file__).parent / "data"
EMBEDDINGS   = DATA_DIR / "item_plm_embeddings.npy"
ITEM_IDS     = DATA_DIR / "item_id_list.pkl"
METADATA     = DATA_DIR / "game_metadata.csv"
RESULTS_FILE = DATA_DIR / "uat_results.csv"
MIN_RATINGS  = 5
TOP_N        = 10

PRIMARY   = "#b81120"
PRIMARY_H = "#dc3135"
SECONDARY = "#0453cd"
TEXT      = "#191c1f"
MUTED     = "#5b5b65"
FAINT     = "#9aa0a8"
BORDER    = "#e2e5ea"
SURFACE   = "#ffffff"
INNER     = "#f2f4f8"
GOLD      = "#f5b400"
GREEN     = "#0f9d58"

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

@st.cache_data
def _logo_b64() -> str:
    return base64.b64encode(_LOGO_PATH.read_bytes()).decode()

st.set_page_config(page_title="Suggestify · Recommendation Engine",
                   page_icon=str(_LOGO_PATH), layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #f7f9fd !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #191c1f !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1200px !important; padding: 1.5rem 2rem 5rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #f2f4f8 !important;
    border-right: 1px solid #e0e2e6 !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem !important; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .35rem !important; }

/* ── Cards ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e2e5ea !important;
    border-radius: 8px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: #ffb3ae !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: .85rem !important;
    border-radius: 4px !important; transition: all .15s !important;
    border: 1px solid #e0e2e6 !important;
    background: #ffffff !important; color: #191c1f !important;
    padding: .5rem 1.25rem !important;
}
.stButton > button:hover { border-color: #b81120 !important; color: #b81120 !important; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #b81120 !important; border-color: #b81120 !important; color: #fff !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #dc3135 !important; border-color: #dc3135 !important;
    box-shadow: 0 4px 14px rgba(184,17,32,.25) !important;
    transform: translateY(-1px) !important; color: #fff !important;
}
.stButton > button:disabled { opacity: .4 !important; transform: none !important; box-shadow:none !important; }
.stFormSubmitButton > button {
    background: #b81120 !important; border: 1px solid #b81120 !important; color: #fff !important;
    font-weight: 600 !important; border-radius: 4px !important;
}
.stFormSubmitButton > button:hover { background: #dc3135 !important; border-color: #dc3135 !important; }

/* ── Text input ── */
.stTextInput > div > div > input {
    background: #ffffff !important; border: 1px solid #e0e2e6 !important;
    border-radius: 4px !important; color: #191c1f !important;
    font-family: 'Inter', sans-serif !important; font-size: .9rem !important;
    padding: .6rem 1rem !important;
}
.stTextInput > div > div > input:focus { border-color: #0453cd !important; box-shadow: 0 0 0 1px #0453cd !important; }
.stTextInput > div > div > input::placeholder { color: #9aa0a8 !important; }
.stTextInput label { color: #5b5b65 !important; font-size: .8rem !important; font-weight: 600 !important; }

/* ── Progress ── */
.stProgress > div > div > div > div { background: #0453cd !important; border-radius: 99px !important; }
[data-testid="stProgressBar"] { background: #e6e8ec !important; border-radius: 99px !important; }

/* ── Star feedback (rating) ── */
[data-testid="stFeedback"] { justify-content: center !important; gap: .1rem !important; }
[data-testid="stFeedback"] * { color: #f5b400 !important; }
[data-testid="stFeedback"] button {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: .1rem !important;
    transform: none !important;
}
[data-testid="stFeedback"] button:hover { transform: scale(1.15) !important; }

/* ── Radio ── */
div[data-baseweb="radio"] label span { color: #191c1f !important; font-size: .875rem !important; }
div[data-baseweb="radio"] label { color: #191c1f !important; }

/* ── SUS horizontal number scale ── */
div[data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] {
    gap: 1.25rem !important; flex-wrap: nowrap !important;
}
div[data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label {
    display: flex !important; align-items: center !important; justify-content: center !important;
    width: 40px !important; height: 40px !important; border-radius: 7px !important;
    border: 1.5px solid #e2e5ea !important; background: #f8f9fc !important;
    cursor: pointer !important; transition: all .15s !important; padding: 0 !important;
}
div[data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:hover {
    border-color: #b81120 !important; background: #fff5f5 !important;
}
div[data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label > div:first-child {
    display: none !important;
}
div[data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label p,
div[data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label span {
    font-size: 1rem !important; font-weight: 700 !important; color: #5b5b65 !important;
    line-height: 1 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important; border: 1px solid #e2e5ea !important; border-radius: 8px !important;
}
details summary { color: #5b5b65 !important; font-size: .82rem !important; font-weight: 600 !important; }
details summary:hover { color: #b81120 !important; }

/* ── Metrics ── */
[data-testid="stMetric"], [data-testid="metric-container"] {
    background: #ffffff !important; border: 1px solid #e2e5ea !important;
    border-radius: 8px !important; padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    color: #5b5b65 !important; font-size: .72rem !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: .06em !important;
}
[data-testid="stMetricValue"] { color: #191c1f !important; font-size: 1.6rem !important; font-weight: 800 !important; }

/* ── Misc ── */
hr { border-color: #e6e8ec !important; margin: 1.5rem 0 !important; }
[data-testid="stAlert"] { border-radius: 8px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f2f4f8; }
::-webkit-scrollbar-thumb { background: #d8dade; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

NAV_ITEMS = [
    ("home",        "Welcome"),
    ("star",        "Rate Products"),
    ("analytics",   "Results"),
    ("quiz",        "Satisfaction Survey"),
    ("bar_chart",   "User Score"),
]

def render_sidebar(step: int):
    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:.65rem;padding:0 .25rem .25rem;margin-bottom:1.1rem;">
            <img src="data:image/png;base64,{_logo_b64()}"
                 style="width:42px;height:42px;object-fit:contain;flex-shrink:0;" alt="Suggestify"/>
            <div>
                <div style="font-size:1.15rem;font-weight:900;color:{PRIMARY};line-height:1.1;letter-spacing:-.02em;">Suggestify</div>
                <div style="font-size:.7rem;color:{MUTED};font-weight:500;">Data-Centric Engine</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("New Analysis", type="primary",
                     use_container_width=True, key="new_analysis"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown(f'<div style="font-size:.68rem;font-weight:700;color:{MUTED};'
                    'text-transform:uppercase;letter-spacing:.1em;padding:1rem .5rem .4rem;">Navigation</div>',
                    unsafe_allow_html=True)

        items_html = ""
        for i, (icon, label) in enumerate(NAV_ITEMS):
            if i == step:
                items_html += (
                    f'<div style="display:flex;align-items:center;padding:.55rem .7rem;'
                    f'border-radius:5px;background:{PRIMARY};color:#fff;font-weight:700;'
                    f'border-left:4px solid #7a0a14;margin-bottom:.2rem;">'
                    f'<span style="font-size:.86rem;">{label}</span></div>')
            elif i < step:
                items_html += (
                    f'<div style="display:flex;align-items:center;padding:.55rem .7rem;'
                    f'border-radius:5px;color:{TEXT};border-left:4px solid transparent;margin-bottom:.2rem;">'
                    f'<span style="font-size:.86rem;font-weight:500;">{label}</span></div>')
            else:
                items_html += (
                    f'<div style="display:flex;align-items:center;padding:.55rem .7rem;'
                    f'border-radius:5px;color:{FAINT};border-left:4px solid transparent;margin-bottom:.2rem;">'
                    f'<span style="font-size:.86rem;font-weight:400;">{label}</span></div>')
        st.markdown(items_html, unsafe_allow_html=True)

        mongo_ok = _mongo_ok()
        dot = GREEN if mongo_ok else PRIMARY
        txt = "MongoDB Connected" if mongo_ok else "Database Offline"
        st.markdown(f"""
        <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e0e2e6;">
            <div style="display:flex;align-items:center;gap:.5rem;padding:.5rem .6rem;
                        background:#ffffff;border:1px solid {BORDER};border-radius:6px;">
                <span style="width:8px;height:8px;border-radius:50%;background:{dot};
                             box-shadow:0 0 0 3px {dot}22;"></span>
                <span style="font-size:.74rem;font-weight:600;color:{TEXT};">{txt}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def page_header(title: str, subtitle: str = ""):
    sub = f'<p style="font-size:.95rem;color:{MUTED};line-height:1.55;max-width:680px;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:1.75rem;">
        <h1 style="font-size:2rem;font-weight:800;color:{TEXT};letter-spacing:-.025em;margin-bottom:.35rem;">{title}</h1>
        {sub}
    </div>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading recommendation model…")
def load_recommender():
    return CBFRecommender(str(EMBEDDINGS), str(ITEM_IDS), str(METADATA))

@st.cache_resource(show_spinner="Seeding database…")
def _auto_seed():
    if db.is_connected() and METADATA.exists():
        db.auto_seed_if_empty(str(METADATA))

def init_state():
    for k, v in {
        "phase": "welcome", "session_id": str(uuid.uuid4())[:8],
        "rated_items": {}, "recommendations": [], "feedback": {},
        "responder_name": "", "start_time": datetime.now().isoformat(),
        "sus_answers": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
_auto_seed()

def go(phase): st.session_state.phase = phase; st.rerun()

def _calc_sus_score(sus_ans: dict) -> float:
    score = 0.0
    for i in range(1, 11):
        v = sus_ans.get(str(i), 0)
        score += (v - 1) * 2.5 if i % 2 == 1 else (5 - v) * 2.5
    return round(score, 1)

def save_result():
    sus_ans   = st.session_state.get("sus_answers", {})
    sus_score = _calc_sus_score(sus_ans)
    row = {
        "session_id":     st.session_state.session_id,
        "responder":      st.session_state.responder_name,
        "timestamp":      st.session_state.start_time,
        "rated_count":    len(st.session_state.rated_items),
        "recs_shown":     len(st.session_state.recommendations),
        "relevant_count": sum(st.session_state.feedback.values()),
        "sus_score":      sus_score,
        "rated_items":    st.session_state.rated_items,
        "feedback":       st.session_state.feedback,
        "sus_answers":    sus_ans,
    }
    if _mongo_ok():
        db.save_uat_result(row)
    else:
        row_csv = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in row.items()}
        df_new = pd.DataFrame([row_csv])
        if RESULTS_FILE.exists():
            pd.concat([pd.read_csv(RESULTS_FILE), df_new], ignore_index=True).to_csv(RESULTS_FILE, index=False)
        else:
            df_new.to_csv(RESULTS_FILE, index=False)

def db_expander():
    mongo_ok = _mongo_ok()
    shard_info, total_items, total_uat = _get_db_stats()
    status_c = GREEN if mongo_ok else PRIMARY
    status_t = "CONNECTED" if mongo_ok else "DISCONNECTED"
    shards   = shard_info.get("shards", [])

    def node(role, name, host, role_c):
        dc = GREEN if mongo_ok else PRIMARY
        return (f'<div style="display:flex;align-items:center;gap:.5rem;padding:.45rem .7rem;'
                f'background:{INNER};border:1px solid {BORDER};border-radius:6px;margin-bottom:.3rem;">'
                f'<div style="width:7px;height:7px;border-radius:50%;background:{dc};flex-shrink:0;"></div>'
                f'<span style="font-size:.64rem;font-weight:700;white-space:nowrap;color:#fff;'
                f'background:{role_c};padding:1px 6px;border-radius:3px;">{role}</span>'
                f'<span style="font-size:.78rem;font-weight:600;color:{TEXT};">{name}</span>'
                f'<span style="font-size:.68rem;color:{FAINT};font-family:monospace;margin-left:auto;">{host}</span>'
                f'</div>')

    nodes_html = node("ROUTER", "mongos", db.MONGO_URI.split("@")[-1].split("/")[0], PRIMARY)
    for s in shards:
        nodes_html += node("SHARD", s.get("id", "?"), s.get("host", "?"), SECONDARY)
    if not shards:
        nodes_html += f'<div style="font-size:.72rem;color:{FAINT};padding:.3rem .7rem;">Info shard tidak tersedia.</div>'

    stat = lambda v, l: (f'<div style="background:{INNER};border:1px solid {BORDER};border-radius:7px;padding:.6rem;text-align:center;">'
                          f'<div style="font-size:1.1rem;font-weight:800;color:{TEXT};">{v}</div>'
                          f'<div style="font-size:.6rem;color:{MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.06em;">{l}</div></div>')

    html = (
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.8rem;">'
        f'<span style="font-size:.82rem;font-weight:700;color:{TEXT};">MongoDB Sharded Cluster</span>'
        f'<span style="font-size:.7rem;font-weight:700;color:{status_c};">● {status_t}</span></div>'
        f'{nodes_html}'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;margin-top:.6rem;">'
        f'{stat(f"{total_items:,}","Items")}{stat(total_uat,"UAT")}{stat(len(shards),"Shards")}</div>'
        f'<div style="margin-top:.6rem;padding-top:.6rem;border-top:1px solid {BORDER};">'
        f'<div style="font-size:.62rem;color:{MUTED};font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.35rem;">Data Distribution</div>'
        f'<div style="font-size:.72rem;color:{MUTED};line-height:1.5;">Aplikasi terhubung melalui mongos router · data terpartisi (sharded) di seluruh shard yang terdaftar.</div>'
        f'</div>'
    )
    with st.expander("System Architecture (Advisor View)"):
        st.markdown(html, unsafe_allow_html=True)

phase = st.session_state.phase

if phase == "welcome":
    render_sidebar(0)

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid {BORDER};border-radius:12px;
                padding:2.5rem 2.75rem;margin-bottom:2.5rem;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-40px;width:240px;height:240px;
                    background:{PRIMARY};opacity:.06;border-radius:50%;filter:blur(40px);"></div>
        <div style="position:relative;max-width:640px;">
            <img src="data:image/png;base64,{_logo_b64()}"
                 style="width:64px;height:64px;object-fit:contain;margin-bottom:1rem;" alt="Suggestify"/>
            <h1 style="font-size:2.4rem;font-weight:800;color:{TEXT};letter-spacing:-.03em;
                       line-height:1.1;margin-bottom:.85rem;">Welcome to Suggestify</h1>
            <p style="font-size:1rem;color:{MUTED};line-height:1.65;">
                Mesin rekomendasi berbasis data yang dirancang untuk presisi. Suggestify menganalisis
                preferensi Anda menggunakan model algoritmik tingkat lanjut untuk menyajikan rekomendasi
                game yang relevan, lalu Anda menilai seberapa baik hasilnya.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<h2 style="font-size:1.3rem;font-weight:700;color:{TEXT};margin-bottom:1.1rem;">'
                'Active Engine Models</h2>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    models = [
        ("01", "ALS-CF", "COLLABORATIVE FILTERING",
         "Matrix factorization Alternating Least Squares pada implicit feedback untuk menemukan pola dari pengguna yang serupa."),
        ("02", "PLM-CBF", "PRE-TRAINED LANGUAGE MODEL",
         "Content-based filtering berbasis Sentence-BERT (all-MiniLM-L6-v2, 384-dim) untuk representasi semantik item."),
        ("03", "GRAM-CF", "PLM + BPR LOSS",
         "PLM item encoder dengan learnable user embedding, dioptimasi menggunakan Bayesian Personalized Ranking."),
    ]
    for col, (num, name, tag, desc) in zip((c1, c2, c3), models):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid {BORDER};border-radius:8px;padding:1.4rem;height:100%;">
                <div style="width:28px;height:28px;background:{PRIMARY}1a;border-radius:6px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:.75rem;font-weight:800;color:{PRIMARY};margin-bottom:.9rem;">{num}</div>
                <div style="font-size:1.05rem;font-weight:700;color:{TEXT};margin-bottom:.25rem;">{name}</div>
                <div style="font-size:.66rem;font-weight:700;color:{MUTED};letter-spacing:.06em;
                            text-transform:uppercase;margin-bottom:.6rem;">{tag}</div>
                <div style="font-size:.84rem;color:{MUTED};line-height:1.55;min-height:3.9rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(f'<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;'
                    f'color:{SECONDARY};margin-bottom:1rem;">Application Workflow</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Rate games you know", "Beri rating bintang pada minimal 5 game yang pernah Anda mainkan."),
            ("2", "We find your matches", "Sistem menganalisis rating Anda dan menyajikan 10 rekomendasi."),
            ("3", "Share your feedback", "Tandai hasil yang relevan, lalu isi survei kegunaan 10 pertanyaan."),
        ]
        steps_html = '<div style="display:flex;flex-direction:column;gap:.6rem;">'
        for num, title, desc in steps:
            steps_html += (
                f'<div style="display:flex;align-items:flex-start;gap:.85rem;padding:.85rem;'
                f'background:#ffffff;border:1px solid {BORDER};border-radius:7px;">'
                f'<div style="width:28px;height:28px;background:{PRIMARY}1a;border-radius:6px;flex-shrink:0;'
                f'display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:800;color:{PRIMARY};">{num}</div>'
                f'<div><div style="font-size:.88rem;font-weight:600;color:{TEXT};">{title}</div>'
                f'<div style="font-size:.78rem;color:{MUTED};margin-top:2px;">{desc}</div></div></div>')
        steps_html += '</div>'
        st.markdown(steps_html, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
                    color:{PRIMARY};margin-bottom:.6rem;">Initialize Session</div>
        <div style="font-size:.85rem;color:{MUTED};margin-bottom:1rem;line-height:1.5;">
            Masukkan nama atau inisial Anda. Respons bersifat anonim dan hanya digunakan
            untuk penelitian akademik.
        </div>
        """, unsafe_allow_html=True)

        name = st.text_input("Nama atau inisial",
                             label_visibility="collapsed")

        if st.button("Start", type="primary",
                     disabled=not name.strip(), use_container_width=True):
            st.session_state.responder_name = name.strip()
            go("onboarding")
        if not name.strip():
            st.markdown(f'<div style="font-size:.78rem;color:{FAINT};margin-top:.5rem;">Masukkan nama untuk melanjutkan</div>',
                        unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1.25rem;padding-top:1rem;border-top:1px solid {BORDER};
                    display:flex;gap:1.1rem;flex-wrap:wrap;">
            <div><div style="font-size:1.15rem;font-weight:800;color:{TEXT};">51K</div>
                 <div style="font-size:.64rem;color:{FAINT};text-transform:uppercase;letter-spacing:.06em;">Games</div></div>
            <div><div style="font-size:1.15rem;font-weight:800;color:{TEXT};">1.6M</div>
                 <div style="font-size:.64rem;color:{FAINT};text-transform:uppercase;letter-spacing:.06em;">Reviews</div></div>
            <div><div style="font-size:1.15rem;font-weight:800;color:{TEXT};">3</div>
                 <div style="font-size:.64rem;color:{FAINT};text-transform:uppercase;letter-spacing:.06em;">Models</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    db_expander()

elif phase == "onboarding":
    render_sidebar(1)
    rec     = load_recommender()
    rated   = st.session_state.rated_items
    n_rated = len(rated)

    hc1, hc2 = st.columns([3, 1])
    with hc1:
        page_header("Rate Products",
                    f"Beri rating bintang pada minimal <strong style='color:{SECONDARY}'>{MIN_RATINGS} game</strong> "
                    "yang pernah Anda mainkan. Semakin banyak yang dinilai, semakin presisi mesin Suggestify.")
    with hc2:
        pct = min(n_rated / MIN_RATINGS * 100, 100)
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid {BORDER};border-radius:8px;padding:.9rem;text-align:center;">
            <div style="font-size:2rem;font-weight:900;color:{TEXT};line-height:1;">{n_rated}</div>
            <div style="font-size:.65rem;color:{MUTED};margin:.25rem 0 .5rem;">rated</div>
            <div style="background:#e6e8ec;border-radius:99px;height:5px;overflow:hidden;">
                <div style="width:{pct:.0f}%;height:100%;background:{SECONDARY};border-radius:99px;"></div>
            </div>
            <div style="font-size:.62rem;color:{FAINT};margin-top:.3rem;">min {MIN_RATINGS}</div>
        </div>""", unsafe_allow_html=True)

    query = st.text_input("", placeholder="  Cari berdasarkan judul…", label_visibility="collapsed")
    games = _search_games(query, n=30) if query.strip() else _get_popular_games(n=60)

    st.markdown("<br>", unsafe_allow_html=True)

    for row_start in range(0, len(games), 3):
        cols = st.columns(3, gap="medium")
        for ci, game in enumerate(games[row_start:row_start + 3]):
            pid   = str(game.get("product_id", ""))
            title = str(game.get("title", pid[:8]))
            img   = game.get("image_url", "")
            cur   = rated.get(pid, 0)
            is_r  = pid in rated

            with cols[ci]:
                with st.container(border=True):
                    if img and str(img).startswith("http"):
                        st.markdown(f'<div style="width:100%;height:145px;overflow:hidden;border-radius:6px;margin-bottom:.65rem;"><img src="{img}" style="width:100%;height:100%;object-fit:cover;display:block;"></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="width:100%;height:145px;background:{INNER};border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:2.5rem;margin-bottom:.65rem;"></div>', unsafe_allow_html=True)

                    t = (title[:44] + "…") if len(title) > 44 else title
                    tc = TEXT if is_r else MUTED
                    label = "Rated" if is_r else "Rate this title"
                    lc    = GOLD if is_r else FAINT
                    st.markdown(
                        f'<div style="font-size:.8rem;font-weight:600;color:{tc};line-height:1.35;min-height:2.2rem;margin-bottom:.5rem;">{t}</div>'
                        f'<div style="font-size:.68rem;font-weight:600;color:{lc};text-align:center;margin-bottom:.15rem;">{label}</div>',
                        unsafe_allow_html=True)

                    fb_key = f"star_{pid}"
                    if cur > 0 and fb_key not in st.session_state:
                        st.session_state[fb_key] = cur - 1
                    sel = st.feedback("stars", key=fb_key)
                    new_rating = (sel + 1) if sel is not None else 0
                    if new_rating != cur:
                        if new_rating == 0: rated.pop(pid, None)
                        else: rated[pid] = new_rating
                        st.session_state.rated_items = rated
                        st.rerun()

    st.divider()
    b1, b2, _ = st.columns([2, 3, 2])
    with b1:
        if st.button("See my recommendations", type="primary",
                     disabled=n_rated < MIN_RATINGS, use_container_width=True):
            go("generating")
    with b2:
        if n_rated < MIN_RATINGS:
            st.markdown(f'<div style="padding-top:.55rem;font-size:.82rem;color:{MUTED};">Rate <strong style="color:{PRIMARY}">{MIN_RATINGS - n_rated} more game{"s" if MIN_RATINGS - n_rated > 1 else ""}</strong> to continue</div>', unsafe_allow_html=True)

elif phase == "generating":
    render_sidebar(2)
    rec = load_recommender()
    st.markdown(f"""
    <div style="text-align:center;padding:5rem 0;">
        <div style="font-size:1.15rem;font-weight:700;color:{TEXT};margin-bottom:.4rem;">Finding your matches</div>
        <div style="font-size:.875rem;color:{MUTED};">Menganalisis rating Anda…</div>
    </div>""", unsafe_allow_html=True)
    with st.spinner(""):
        time.sleep(1)
        recs = rec.recommend(st.session_state.rated_items, top_n=TOP_N)
    st.session_state.recommendations = recs
    go("recommendations")

elif phase == "recommendations":
    render_sidebar(2)
    recs     = st.session_state.recommendations
    feedback = st.session_state.feedback

    page_header("Top 10 Recommendations",
                "Berdasarkan rating dan riwayat interaksi Anda. Untuk setiap game, "
                "tandai apakah sesuai dengan selera Anda, jawab semua 10 untuk lanjut.")

    if not recs:
        st.warning("No recommendations generated. Try rating more games.")
        if st.button("Go back"): go("onboarding")
    else:
        answered = len(feedback)
        total    = len(recs)
        done_all = answered == total

        rc1, rc2 = st.columns([4, 1])
        with rc1:
            st.progress(answered / total)
        with rc2:
            st.markdown(f'<div style="text-align:right;font-size:.8rem;color:{GREEN if done_all else MUTED};padding-top:.15rem;">{answered}/{total} answered</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        scores  = [g["score"] for g in recs]
        s_min, s_rng = min(scores), max(max(scores) - min(scores), 1e-6)

        for i, game in enumerate(recs, 1):
            pid     = game["product_id"]
            title   = game["title"]
            img     = game.get("image_url", "")
            score   = game["score"]
            pct     = int(((score - s_min) / s_rng) * 55 + 35)
            pct     = min(max(pct, 20), 99)
            rc      = game.get("rating_count", 0)
            p_url   = game.get("product_url", f"https://www.amazon.com/dp/{pid}")
            cur_fb  = feedback.get(pid)

            with st.container(border=True):
                c0, c1, c2, c3 = st.columns([1, 2, 5, 3], gap="medium")
                with c0:
                    st.markdown(f'<div style="margin-top:.4rem;width:32px;height:32px;background:{INNER};border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:.82rem;font-weight:800;color:{PRIMARY};">#{i}</div>', unsafe_allow_html=True)
                with c1:
                    if img and str(img).startswith("http"):
                        st.markdown(f'<div style="width:100%;height:82px;overflow:hidden;border-radius:6px;"><img src="{img}" style="width:100%;height:100%;object-fit:cover;"></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="width:100%;height:82px;background:{INNER};border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;"></div>', unsafe_allow_html=True)
                with c2:
                    rc_txt = f'<div style="font-size:.72rem;color:{FAINT};margin-bottom:.45rem;">{int(rc):,} reviews</div>' if rc and int(rc) > 0 else ""
                    st.markdown(
                        f'<div style="font-size:.9rem;font-weight:700;color:{TEXT};line-height:1.3;margin-bottom:.25rem;">{title}</div>'
                        f'{rc_txt}'
                        f'<div style="font-size:.68rem;color:{MUTED};margin-bottom:.25rem;">Match score <span style="color:{SECONDARY};font-weight:700;">{pct}%</span></div>'
                        f'<div style="background:#e6e8ec;border-radius:99px;height:5px;overflow:hidden;margin-bottom:.6rem;">'
                        f'<div style="width:{pct}%;height:100%;background:{SECONDARY};border-radius:99px;"></div></div>'
                        f'<a href="{p_url}" target="_blank" style="font-size:.78rem;font-weight:600;color:{SECONDARY};'
                        f'text-decoration:none;white-space:nowrap;">View Details</a>',
                        unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div style="font-size:.75rem;color:{MUTED};margin-bottom:.4rem;">Is this relevant?</div>',
                                unsafe_allow_html=True)
                    tcol1, tcol2 = st.columns(2)
                    with tcol1:
                        if st.button("", icon=":material/thumb_up:",
                                     key=f"up_{pid}", use_container_width=True,
                                     type=("primary" if cur_fb is True else "secondary")):
                            feedback[pid] = True
                            st.session_state.feedback = feedback
                            st.rerun()
                    with tcol2:
                        if st.button("", icon=":material/thumb_down:",
                                     key=f"dn_{pid}", use_container_width=True,
                                     type=("primary" if cur_fb is False else "secondary")):
                            feedback[pid] = False
                            st.session_state.feedback = feedback
                            st.rerun()

        st.session_state.feedback = feedback
        st.divider()
        sb, _ = st.columns([2, 4])
        with sb:
            if st.button("Lanjut ke Survei", type="primary",
                         disabled=not done_all, use_container_width=True):
                go("sus_survey")
        if not done_all:
            st.markdown(f'<div style="font-size:.78rem;color:{MUTED};margin-top:.5rem;">{total - answered} answer{"s" if total-answered>1 else ""} remaining</div>', unsafe_allow_html=True)

elif phase == "sus_survey":
    render_sidebar(3)

    SUS_QUESTIONS = [
        ("Saya pikir saya akan ingin menggunakan sistem ini secara rutin.",
         "Evaluasi niat penggunaan berkelanjutan."),
        ("Saya merasa sistem ini terlalu rumit.",
         "Kompleksitas antarmuka dan alur kerja."),
        ("Saya pikir sistem ini mudah digunakan.",
         "Kemudahan navigasi dan interaksi."),
        ("Saya pikir saya perlu bantuan orang teknis untuk menggunakan sistem ini.",
         "Kebutuhan dukungan eksternal."),
        ("Saya menemukan berbagai fungsi dalam sistem ini terintegrasi dengan baik.",
         "Koherensi fitur dan integrasi antar halaman."),
        ("Saya pikir ada terlalu banyak inkonsistensi dalam sistem ini.",
         "Konsistensi tampilan dan perilaku antarmuka."),
        ("Saya bayangkan kebanyakan orang akan sangat cepat mempelajari sistem ini.",
         "Kemudahan dipelajari oleh pengguna baru."),
        ("Saya merasa sistem ini sangat berat untuk digunakan.",
         "Hambatan dan gesekan saat berinteraksi."),
        ("Saya merasa sangat percaya diri menggunakan sistem ini.",
         "Kepercayaan diri selama sesi berlangsung."),
        ("Saya perlu banyak belajar hal sebelum bisa menggunakan sistem ini.",
         "Kurva pembelajaran sistem secara keseluruhan."),
    ]

    sus_ans = st.session_state.sus_answers

    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2rem;font-weight:800;color:{TEXT};letter-spacing:-.025em;margin-bottom:.35rem;">Satisfaction Survey</h1>
        <p style="font-size:.95rem;color:{MUTED};line-height:1.55;max-width:680px;">
            Isian Anda membantu mengkalibrasi mesin rekomendasi Suggestify.
            Harap luangkan waktu untuk mengevaluasi sesi Anda.
        </p>
    </div>
    <div style="display:flex;align-items:center;gap:2rem;margin-bottom:1.5rem;padding:.65rem 1.25rem;
                background:#ffffff;border:1px solid {BORDER};border-radius:8px;font-size:.78rem;color:{MUTED};">
        <span><strong style="color:{TEXT};">1</strong> = Sangat Tidak Setuju</span>
        <span style="color:{BORDER};">|</span>
        <span><strong style="color:{TEXT};">2</strong> = Tidak Setuju</span>
        <span style="color:{BORDER};">|</span>
        <span><strong style="color:{TEXT};">3</strong> = Netral</span>
        <span style="color:{BORDER};">|</span>
        <span><strong style="color:{TEXT};">4</strong> = Setuju</span>
        <span style="color:{BORDER};">|</span>
        <span><strong style="color:{TEXT};">5</strong> = Sangat Setuju</span>
    </div>""", unsafe_allow_html=True)

    with st.container(border=True):
        for pair in range(0, 10, 2):
            col_l, col_r = st.columns(2, gap="large")
            for ci, qi in enumerate(range(pair, pair + 2)):
                q_text, q_sub = SUS_QUESTIONS[qi]
                q_num = qi + 1
                cur_val = sus_ans.get(str(q_num))
                cur_idx = (cur_val - 1) if cur_val else None
                with (col_l if ci == 0 else col_r):
                    st.markdown(
                        f'<div style="font-size:.9rem;font-weight:700;color:{TEXT};'
                        f'margin-bottom:.15rem;line-height:1.4;">{q_num}. {q_text}</div>'
                        f'<div style="font-size:.72rem;color:{MUTED};margin-bottom:.6rem;">{q_sub}</div>',
                        unsafe_allow_html=True)
                    sel = st.radio("", ["1", "2", "3", "4", "5"], index=cur_idx,
                                   key=f"sus_q{q_num}", horizontal=True,
                                   label_visibility="collapsed")
                    if sel:
                        sus_ans[str(q_num)] = int(sel)
            if pair < 8:
                st.markdown(
                    '<div style="border-top:1px solid #f0f2f5;margin:.75rem 0;"></div>',
                    unsafe_allow_html=True)

    st.session_state.sus_answers = sus_ans
    all_answered = len(sus_ans) == 10

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col = st.columns([3, 1])
    with btn_col:
        if st.button("Submit Survey", type="primary",
                     disabled=not all_answered, use_container_width=True):
            save_result()
            go("user_score")
    if not all_answered:
        remaining = 10 - len(sus_ans)
        st.markdown(
            f'<div style="font-size:.78rem;color:{MUTED};margin-top:.5rem;text-align:right;">'
            f'{remaining} pertanyaan belum dijawab</div>', unsafe_allow_html=True)

elif phase == "user_score":
    render_sidebar(4)

    sus_ans      = st.session_state.get("sus_answers", {})
    sus_score    = _calc_sus_score(sus_ans)
    relevant     = sum(1 for v in st.session_state.feedback.values() if v)
    shown        = len(st.session_state.recommendations)
    rated        = len(st.session_state.rated_items)
    accuracy_pct = int(relevant / shown * 100) if shown else 0

    if rated >= 16:   depth_level, depth_label = 5, "Expert"
    elif rated >= 12: depth_level, depth_label = 4, "High"
    elif rated >= 9:  depth_level, depth_label = 3, "Moderate"
    elif rated >= 7:  depth_level, depth_label = 2, "Basic"
    else:             depth_level, depth_label = 1, "Minimal"

    if sus_score >= 80:
        s_color, s_grade = GREEN,     "Excellent"
        s_desc = ("Sesi dengan fidelitas tinggi. Respons Anda menunjukkan pola yang konsisten "
                  "dan mengarah pada rekomendasi yang sangat akurat.")
    elif sus_score >= 68:
        s_color, s_grade = SECONDARY, "Good"
        s_desc = ("Sesi yang baik. Sistem cukup mudah digunakan dan memberikan pengalaman yang positif.")
    else:
        s_color, s_grade = PRIMARY,   "Needs Improvement"
        s_desc = ("Ada area yang perlu ditingkatkan. Umpan balik Anda akan membantu kami memperbaiki sistem.")

    r      = 75
    circ   = 2 * 3.14159 * r
    dash   = circ * sus_score / 100

    st.markdown(f"""
    <div style="margin-bottom:1.75rem;">
        <h1 style="font-size:2rem;font-weight:800;color:{TEXT};letter-spacing:-.025em;margin-bottom:.35rem;">Analysis Profile</h1>
        <p style="font-size:.95rem;color:{MUTED};line-height:1.55;max-width:680px;">
            Metrik interaksi dan ringkasan kualitas sesi Anda.
        </p>
    </div>""", unsafe_allow_html=True)

    score_col, status_col = st.columns([3, 2], gap="medium")

    with score_col:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid {BORDER};border-radius:12px;
                    padding:2rem;position:relative;overflow:hidden;min-height:210px;">
            <div style="position:absolute;right:-40px;top:50%;transform:translateY(-50%);opacity:.18;">
                <svg width="220" height="220" viewBox="0 0 200 200">
                    <circle cx="100" cy="100" r="{r}" fill="none" stroke="{s_color}" stroke-width="22"/>
                    <circle cx="100" cy="100" r="{r}" fill="none" stroke="#ffffff" stroke-width="22"
                            stroke-dasharray="{circ - dash:.1f} {circ:.1f}"
                            transform="rotate(-90 100 100)"/>
                </svg>
            </div>
            <div style="position:relative;z-index:1;">
                <div style="font-size:.62rem;font-weight:700;letter-spacing:.1em;
                            text-transform:uppercase;color:{MUTED};margin-bottom:1rem;">USER EXPERIENCE SCORE</div>
                <div style="display:flex;align-items:baseline;gap:.4rem;margin-bottom:.6rem;">
                    <span style="font-size:3.5rem;font-weight:900;color:{s_color};line-height:1;">{sus_score:.0f}</span>
                    <span style="font-size:1rem;color:{MUTED};font-weight:600;">/100</span>
                </div>
                <div style="display:inline-block;font-size:.7rem;font-weight:700;color:{s_color};
                            background:{s_color}1a;padding:.25rem .75rem;border-radius:4px;margin-bottom:.75rem;">{s_grade}</div>
                <div style="font-size:.84rem;color:{MUTED};line-height:1.6;max-width:380px;">{s_desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with status_col:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid {BORDER};border-radius:12px;
                    padding:2rem;min-height:210px;">
            <div style="font-size:.62rem;font-weight:700;letter-spacing:.1em;
                        text-transform:uppercase;color:{MUTED};margin-bottom:1rem;">SESSION STATUS</div>
            <div style="display:flex;align-items:center;margin-bottom:.75rem;">
                <span style="font-size:1.25rem;font-weight:800;color:{SECONDARY};">Complete</span>
            </div>
            <div style="font-size:.83rem;color:{MUTED};line-height:1.6;margin-bottom:1.25rem;">
                Semua node data yang diperlukan telah terisi untuk model saat ini.
            </div>
            <div style="font-size:.68rem;color:{MUTED};margin-bottom:.3rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;">Session ID</div>
            <div style="font-family:monospace;font-size:.8rem;color:{PRIMARY};font-weight:600;
                        background:{PRIMARY}0f;padding:.35rem .65rem;border-radius:4px;">
                {st.session_state.session_id}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Restart Session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<h2 style="font-size:1.2rem;font-weight:700;color:{TEXT};margin-bottom:1rem;">Metrics Breakdown</h2>',
                unsafe_allow_html=True)

    if accuracy_pct >= 80:   acc_badge, acc_bc = "Excellent", GREEN
    elif accuracy_pct >= 60: acc_badge, acc_bc = "Good",      SECONDARY
    else:                    acc_badge, acc_bc = "Moderate",  "#f29900"

    def _metric_card(icon, label, value, badge, bc, bar_pct, bar_c):
        return (
            f'<div style="background:#ffffff;border:1px solid {BORDER};border-radius:10px;'
            f'padding:1.4rem;height:100%;">'
            f'<div style="display:flex;align-items:flex-start;justify-content:flex-end;margin-bottom:1rem;">'
            f'<span style="font-size:.7rem;font-weight:700;color:{bc};">{badge}</span></div>'
            f'<div style="font-size:.6rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
            f'color:{MUTED};margin-bottom:.35rem;">{label}</div>'
            f'<div style="font-size:1.7rem;font-weight:800;color:{TEXT};margin-bottom:.85rem;">{value}</div>'
            f'<div style="background:#e6e8ec;border-radius:99px;height:5px;overflow:hidden;">'
            f'<div style="width:{bar_pct}%;height:100%;background:{bar_c};border-radius:99px;"></div></div>'
            f'</div>')

    mc1, mc2, mc3 = st.columns(3, gap="medium")
    with mc1:
        st.markdown(_metric_card("adjust",    "RATING ACCURACY", f"{accuracy_pct}%",
                                 acc_badge, acc_bc, accuracy_pct, SECONDARY), unsafe_allow_html=True)
    with mc2:
        st.markdown(_metric_card("database",  "PROFILE DEPTH",   f"Level {depth_level}",
                                 depth_label, PRIMARY, depth_level / 5 * 100, PRIMARY), unsafe_allow_html=True)
    with mc3:
        st.markdown(_metric_card("checklist", "SURVEY COMPLETION", "100%",
                                 "Done", TEXT, 100, TEXT), unsafe_allow_html=True)
