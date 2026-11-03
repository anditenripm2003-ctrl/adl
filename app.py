import base64
import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sparse
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommender Dashboard", page_icon="📖", layout="wide",
                   initial_sidebar_state="expanded")

BASE = Path(__file__).resolve().parent

# Palet: hijau hutan, emas, krem kertas, burgundy (nuansa "reading nook")
FOREST, PINE, SAGE, MINT = "#0f3d2e", "#1f6b4f", "#8fb59a", "#dfeadf"
GOLD, AMBER, BURGUNDY, CREAM = "#e3a83b", "#f3c969", "#8c2f39", "#fbf6ec"
INK, MUTED, LINE = "#1d2b24", "#6f7d74", "#e9dfcc"


# =========================================================
# Gaya
# =========================================================
def hero_bg():
    p = BASE / "hero_books.jpg"
    if not p.exists():
        return f"linear-gradient(120deg,{CREAM},#f3e6cc)"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f"linear-gradient(90deg,{CREAM} 0%,{CREAM} 62%,rgba(251,246,236,.92) 71%,rgba(251,246,236,.4) 79%,rgba(251,246,236,0) 88%),"
            f"url(data:image/jpeg;base64,{b64}) right center/auto 100% no-repeat,{CREAM}")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;0,9..144,800;1,9..144,600;1,9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--forest:#0f3d2e;--pine:#1f6b4f;--sage:#8fb59a;--gold:#e3a83b;--amber:#f3c969;--burgundy:#8c2f39;
  --cream:#fbf6ec;--ink:#1d2b24;--muted:#6f7d74;--line:#e9dfcc;}
html,body,[class*="css"],[data-testid="stAppViewContainer"],button,input,textarea,select{font-family:'Plus Jakarta Sans',sans-serif;}
.stApp{background:radial-gradient(1200px 600px at 100% -10%,#f1e6cf 0%,rgba(241,230,207,0) 60%),
  radial-gradient(900px 500px at -10% 110%,#e3eee2 0%,rgba(227,238,226,0) 60%),linear-gradient(160deg,#fcf8f0,#f6f1e6);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.3rem;padding-bottom:3rem;max-width:1380px;}
h1,h2,h3{font-family:'Fraunces',serif !important;color:var(--forest);}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(185deg,#0b3326 0%,#13503b 55%,#1f6b4f 100%);}
[data-testid="stSidebar"] *{color:#e7efe8;}
[data-testid="stSidebar"] button{justify-content:flex-start;border:1px solid rgba(255,255,255,.08);border-radius:14px;
  padding:.62rem 1rem;font-weight:600;box-shadow:none;transition:all .18s ease;}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:rgba(255,255,255,.04);}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover{background:rgba(255,255,255,.12);transform:translateX(3px);border-color:rgba(243,201,105,.4);}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]{background:linear-gradient(90deg,#f3c969,#e3a83b);border:0;
  box-shadow:0 8px 20px rgba(227,168,59,.35);}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p{color:#0f3d2e !important;font-weight:800;}
.logo{display:flex;gap:.7rem;align-items:center;margin:.2rem 0 1.6rem;}
.logo .ic{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,#f3c969,#e3a83b);display:flex;
  align-items:center;justify-content:center;font-size:1.4rem;box-shadow:0 6px 16px rgba(0,0,0,.25);}
.logo b{font-family:'Fraunces',serif;font-size:1.32rem;line-height:1.05;color:#fff !important;display:block;}
.logo small{font-size:.66rem;letter-spacing:.18em;color:#f3c969 !important;font-weight:700;}
.side-note{margin-top:1.6rem;padding:1rem;border-radius:16px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
  font-size:.8rem;line-height:1.5;}
.side-note b{color:#f3c969 !important;}

/* ---------- Hero ---------- */
.st-key-hero{background:__HERO__;border-radius:28px;padding:2.4rem 2.8rem 2.4rem;overflow:hidden;position:relative;
  box-shadow:0 18px 40px rgba(15,61,46,.12);border:1px solid #efe4cf;}
.st-key-hero:after{content:"";position:absolute;left:0;right:0;bottom:0;height:8px;background:linear-gradient(90deg,#0f3d2e,#1f6b4f,#e3a83b);}
.st-key-hero>div{max-width:58%;position:relative;z-index:2;}
.pill{display:inline-flex;gap:.45rem;align-items:center;background:#fff;border:1px solid var(--line);border-radius:40px;
  padding:.35rem 1rem;color:var(--forest);font-size:.86rem;font-weight:600;box-shadow:0 4px 12px rgba(15,61,46,.06);}
.hl{font-family:'Fraunces',serif;font-size:clamp(2.4rem,4.4vw,4rem);line-height:1.02;margin:1rem 0 .6rem;color:var(--forest);font-weight:800;}
.hl em{font-style:italic;font-weight:700;background:linear-gradient(90deg,#c98a1c,#e3a83b,#8c2f39);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.sub{color:#3b4b43;font-size:1.02rem;line-height:1.6;max-width:470px;margin:0 0 1rem;}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.2rem 0 1.1rem;}
.chip{background:rgba(255,255,255,.85);border:1px solid var(--line);border-radius:12px;padding:.4rem .8rem;font-size:.8rem;color:var(--muted);}
.chip b{color:var(--forest);font-size:.95rem;margin-right:.25rem;}
.st-key-hero button{background:linear-gradient(90deg,#0f3d2e,#1f6b4f);border:0;border-radius:50px;padding:.7rem 2.2rem;
  box-shadow:0 10px 22px rgba(15,61,46,.3);transition:transform .15s;}
.st-key-hero button:hover{transform:translateY(-2px);}
.st-key-hero button p{color:#fff !important;font-size:1rem;font-weight:700;}

/* ---------- Heading seksi ---------- */
.sec{margin:2.1rem 0 .9rem;}
.sec .k{font-size:.72rem;font-weight:800;letter-spacing:.16em;color:var(--burgundy);text-transform:uppercase;}
.sec b{display:block;font-family:'Fraunces',serif;font-size:1.75rem;font-weight:800;color:var(--forest);line-height:1.15;}
.sec span{color:var(--muted);font-size:.9rem;}
.page-title{font-family:'Fraunces',serif;font-size:2.4rem;font-weight:800;color:var(--forest);margin:.2rem 0 .1rem;}
.page-sub{color:var(--muted);margin-bottom:1rem;}

/* ---------- Kontainer / kartu ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(255,255,255,.78);border-radius:20px;border-color:var(--line) !important;
  box-shadow:0 6px 20px rgba(29,43,36,.05);transition:box-shadow .2s, transform .2s, border-color .2s;}
[data-testid="stVerticalBlockBorderWrapper"]:hover{box-shadow:0 14px 30px rgba(15,61,46,.10);border-color:#cfdccf !important;}
.bk{text-align:center;position:relative;}
.cv{position:relative;width:100%;max-width:120px;margin:0 auto;}
.cover{width:100%;height:170px;object-fit:cover;border-radius:6px 12px 12px 6px;display:block;
  box-shadow:-4px 0 0 rgba(0,0,0,.08) inset,0 10px 22px rgba(15,61,46,.25);transition:transform .25s;background:#efe7d6;}
.bk:hover .cover{transform:translateY(-4px) rotate(-1deg);}
.ph{display:flex;align-items:center;justify-content:center;text-align:center;background:linear-gradient(160deg,#0f3d2e,#1f6b4f 60%,#8c2f39);
  color:#fbf6ec;font-family:'Fraunces',serif;font-weight:700;font-size:.8rem;padding:.6rem;box-sizing:border-box;}
.rank{position:absolute;top:-8px;left:-8px;width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#f3c969,#e3a83b);
  color:#0f3d2e;font-weight:800;font-size:.8rem;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 10px rgba(0,0,0,.2);z-index:3;}
.bt{font-weight:800;font-size:.88rem;line-height:1.25;margin-top:.7rem;min-height:2.5em;color:var(--forest);
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.ba{color:var(--muted);font-size:.76rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.strow{display:flex;gap:.3rem;justify-content:center;align-items:center;margin-top:.3rem;font-size:.74rem;color:var(--muted);}
.stars{font-size:.95rem;letter-spacing:1px;line-height:1;background:linear-gradient(90deg,#e3a83b var(--p),#ddd5c4 var(--p));
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.strow b{color:var(--ink);font-size:.8rem;}
.match{margin-top:.45rem;font-size:.7rem;color:var(--pine);font-weight:700;}
.mbar{height:5px;border-radius:5px;background:#ece5d6;margin:.25rem auto 0;max-width:120px;overflow:hidden;}
.mbar i{display:block;height:100%;background:linear-gradient(90deg,#1f6b4f,#e3a83b);border-radius:5px;}
.why{font-size:.68rem;color:var(--muted);margin-top:.3rem;line-height:1.3;min-height:2.6em;}
.why b{color:var(--burgundy);font-weight:700;}

/* ---------- KPI ---------- */
.kpi{background:linear-gradient(150deg,#ffffff,#fbf6ec);border:1px solid var(--line);border-radius:20px;padding:1rem 1.1rem;
  display:flex;gap:.85rem;align-items:center;box-shadow:0 6px 18px rgba(29,43,36,.05);height:100%;}
.kpi .i{flex:0 0 46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.25rem;}
.kpi .l{font-size:.74rem;color:var(--muted);font-weight:600;}
.kpi .v{font-family:'Fraunces',serif;font-size:1.55rem;font-weight:800;color:var(--forest);line-height:1.1;}
.kpi .s{font-size:.7rem;color:var(--muted);}
.g1{background:linear-gradient(135deg,#0f3d2e,#1f6b4f);} .g2{background:linear-gradient(135deg,#f3c969,#e3a83b);}
.g3{background:linear-gradient(135deg,#a8424d,#8c2f39);} .g4{background:linear-gradient(135deg,#a9cdb2,#6f9f80);}

/* ---------- Info / langkah ---------- */
.step{border-radius:20px;padding:1.2rem 1.2rem;height:100%;border:1px solid var(--line);background:#fff;position:relative;overflow:hidden;}
.step:before{content:"";position:absolute;top:0;left:0;right:0;height:5px;background:var(--c);}
.step .n{font-size:.7rem;font-weight:800;letter-spacing:.14em;color:var(--c);}
.step b{display:block;font-family:'Fraunces',serif;font-size:1.15rem;color:var(--forest);margin:.25rem 0 .35rem;}
.step p{font-size:.84rem;color:#4d5c54;margin:0;line-height:1.5;}
.note{background:linear-gradient(120deg,#fff8e8,#fdf2dc);border:1px solid #f0dcae;border-left:5px solid var(--gold);
  border-radius:14px;padding:.85rem 1rem;font-size:.86rem;color:#4a3f26;margin:.4rem 0 .8rem;}
.note.g{background:linear-gradient(120deg,#eef6ef,#e3efe4);border-color:#cfe1d2;border-left-color:var(--pine);color:#21412f;}
.ct{font-weight:800;color:var(--forest);font-size:1rem;margin-bottom:.1rem;}
.cs{color:var(--muted);font-size:.78rem;margin-bottom:.3rem;}
.ins{font-size:.8rem;color:#4d5c54;border-top:1px dashed var(--line);padding-top:.55rem;margin-top:.2rem;}
.ins b{color:var(--burgundy);}

/* ---------- Penulis ---------- */
.au{text-align:center;}
.av{width:68px;height:68px;border-radius:50%;margin:.3rem auto .5rem;display:flex;align-items:center;justify-content:center;
  font-family:'Fraunces',serif;font-weight:800;font-size:1.35rem;color:#fff;background:linear-gradient(135deg,#1f6b4f,#0f3d2e);
  box-shadow:0 0 0 4px #fff,0 0 0 6px #e3a83b;}
.au .rk{font-size:.68rem;font-weight:800;color:var(--burgundy);letter-spacing:.1em;}
.au .nm{font-weight:800;color:var(--forest);font-size:.88rem;min-height:2.4em;line-height:1.2;}
.au .sm{font-size:.74rem;color:var(--muted);}

/* ---------- Riwayat ---------- */
.row{display:flex;gap:.7rem;align-items:center;padding:.45rem 0;border-bottom:1px dashed var(--line);}
.row:last-child{border-bottom:0;}
.row img,.row .ph{width:38px;height:56px;border-radius:4px 7px 7px 4px;object-fit:cover;flex:0 0 38px;font-size:.5rem;padding:.2rem;}
.row b{font-size:.84rem;display:block;color:var(--forest);line-height:1.2;}
.row small{color:var(--muted);font-size:.74rem;}
.tag{display:inline-block;background:#f3ecdc;color:var(--burgundy);border-radius:20px;padding:.1rem .55rem;font-size:.7rem;font-weight:700;}

/* ---------- Widget ---------- */
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid #ddd2bb;background:#fff;color:var(--forest);font-weight:700;
  padding:.3rem .7rem;min-height:0;transition:all .15s;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--forest);background:var(--forest);color:#fff;}
.stButton>button:hover p{color:#fff;}
[data-testid="stTextInput"] input{border-radius:14px;background:#fff;border:1px solid var(--line);padding:.7rem 1rem;}
[data-baseweb="select"]>div{border-radius:14px !important;background:#fff !important;border-color:var(--line) !important;}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;background:rgba(255,255,255,.6);padding:.35rem;border-radius:16px;border:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{border-radius:12px;padding:.45rem 1rem;font-weight:700;height:auto;}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,#0f3d2e,#1f6b4f);}
.stTabs [aria-selected="true"] p{color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
[data-testid="stPlotlyChart"]{border-radius:14px;overflow:hidden;}
[data-testid="stExpander"]{border-radius:16px;border-color:var(--line);background:rgba(255,255,255,.6);}
div[role="dialog"]{border-radius:24px;}
/* slider tetap bertema meski .streamlit/config.toml belum ada */
[data-testid="stSlider"] [role="slider"]{background:#1f6b4f !important;border:3px solid #fff !important;box-shadow:0 2px 8px rgba(15,61,46,.35) !important;}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],[data-testid="stSlider"] [data-testid="stThumbValue"]{color:#1f6b4f !important;font-weight:700;}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child{background:linear-gradient(90deg,#1f6b4f,#e3a83b) !important;}
[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"]{color:#6f7d74 !important;}
[data-testid="stFormSubmitButton"] button{background:linear-gradient(90deg,#0f3d2e,#1f6b4f) !important;border:0 !important;border-radius:14px;padding:.6rem 1rem;}
[data-testid="stFormSubmitButton"] button p{color:#fff !important;font-weight:800;}
[data-testid="stForm"]{background:rgba(255,255,255,.78);border-radius:20px;border-color:var(--line) !important;}
span[data-baseweb="tag"]{background:#1f6b4f !important;border-radius:10px !important;}
span[data-baseweb="tag"] span{color:#fff !important;}
</style>
"""
st.markdown(CSS.replace("__HERO__", hero_bg()), unsafe_allow_html=True)


# =========================================================
# Data
# =========================================================
NEED = ["cf_model.joblib", "tfidf_matrix.npz", "model_ratings.pkl", "candidate_books.pkl",
        "cf_comparison.csv", "alpha_evaluation.csv", "metadata.json"]


@st.cache_resource(show_spinner="Memuat model dan data...")
def load():
    miss = [f for f in NEED if not (BASE / f).exists()]
    if miss:
        raise FileNotFoundError("File belum ada: " + ", ".join(miss))
    D = {"cf": joblib.load(BASE / "cf_model.joblib"), "tfidf": sparse.load_npz(BASE / "tfidf_matrix.npz").tocsr(),
         "ratings": pd.read_pickle(BASE / "model_ratings.pkl"), "books": pd.read_pickle(BASE / "candidate_books.pkl"),
         "cmp": pd.read_csv(BASE / "cf_comparison.csv"), "alpha": pd.read_csv(BASE / "alpha_evaluation.csv"),
         "meta": json.load(open(BASE / "metadata.json", encoding="utf-8"))}
    r, b = D["ratings"], D["books"].reset_index(drop=True)
    r["User-ID"] = pd.to_numeric(r["User-ID"], errors="coerce").astype("Int64")
    r["Book-Rating"] = pd.to_numeric(r["Book-Rating"], errors="coerce")
    r["ISBN"], b["ISBN"] = r["ISBN"].astype(str), b["ISBN"].astype(str)
    D["books"] = b
    D["row"] = {x: i for i, x in enumerate(b["ISBN"])}

    # Statistik rating umum (seluruh rating eksplisit 1-10 Book-Crossing) -> bintang 1-5
    bs_path = BASE / "book_stats.csv"
    if bs_path.exists():
        bs = pd.read_csv(bs_path, dtype={"ISBN": str})
    else:  # cadangan: hitung dari rating model
        g = r.groupby("ISBN")["Book-Rating"].agg(n="size", avg="mean").reset_index()
        C = r["Book-Rating"].mean()
        g["bayes"] = g.n / (g.n + 10) * g.avg + 10 / (g.n + 10) * C
        bs = g
    meta_cols = ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-M", "Image-URL-L"]
    cat = pd.concat([b[meta_cols], bs[[c for c in meta_cols if c in bs.columns]]]).drop_duplicates("ISBN")
    cat = cat.merge(bs[["ISBN", "n", "avg", "bayes"]], on="ISBN", how="left")
    cat["n"] = cat["n"].fillna(0).astype(int)
    cat["Year-Of-Publication"] = pd.to_numeric(cat["Year-Of-Publication"], errors="coerce")
    cat["in_model"] = cat["ISBN"].isin(D["row"])
    D["cat"] = cat
    D["bidx"] = cat.set_index("ISBN")

    try:
        D["vec"] = joblib.load(BASE / "tfidf_vectorizer.joblib") if (BASE / "tfidf_vectorizer.joblib").exists() else None
    except Exception:
        D["vec"] = None
    D["eda"] = json.load(open(BASE / "eda_summary.json", encoding="utf-8")) if (BASE / "eda_summary.json").exists() else None
    if D["eda"] is not None and "lorenz" not in D["eda"]:
        D["eda"] = None  # format lama: jalankan ulang prep_eda.py
    D["eda_books"] = (pd.read_csv(BASE / "eda_books.csv", dtype={"ISBN": str}) if (BASE / "eda_books.csv").exists()
                      else pd.DataFrame(columns=["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication",
                                                 "Interaksi", "Rating eksplisit", "Rata-rata", "Skor berbobot"]))
    D["cu"] = pd.read_csv(BASE / "cluster_users.csv") if (BASE / "cluster_users.csv").exists() else None
    D["pb"] = pd.read_csv(BASE / "popular_books.csv", dtype={"ISBN": str}) if (BASE / "popular_books.csv").exists() else None

    cnt = r.groupby("User-ID").size()
    D["ucount"] = cnt.to_dict()
    D["uids"] = cnt.sort_values(ascending=False).index.astype(int).tolist()
    return D


try:
    D = load()
except Exception as exc:
    st.error(f"Dashboard belum bisa dimuat: {exc}")
    st.stop()

CF, TFIDF, RATINGS, BOOKS, META, CAT = D["cf"], D["tfidf"], D["ratings"], D["books"], D["meta"], D["cat"]
BEST_ALPHA = float(META.get("best_alpha", 0.1))
LIKE_THR = int(META.get("like_threshold", 8))
ss = st.session_state
ss.setdefault("page", "Beranda")
ss.setdefault("author", None)
ss.setdefault("user", D["uids"][0])


def num(x, d=0):
    """Format angka gaya Indonesia: 1.234.567 / 7,60"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def short(x):
    if x >= 1e6:
        return num(x / 1e6, 2) + " jt"
    if x >= 1e3:
        return num(x / 1e3, 1) + " rb"
    return num(x)


# =========================================================
# Logika rekomendasi
# =========================================================
def norm_title(t):
    import re
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", str(t).lower())
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def dedupe(df, exclude_isbns=(), n=None):
    """Buang edisi ganda (judul sama, ISBN beda) dan judul yang sudah dibaca/dipilih."""
    seen = {norm_title(D["bidx"].loc[i, "Book-Title"]) for i in exclude_isbns if i in D["bidx"].index}
    keys = df["Book-Title"].map(norm_title)
    keep = ~keys.duplicated() & ~keys.isin(seen)
    out = df[keep.to_numpy()]
    return out.head(n) if n else out


def liked_of(hist, thr):
    hist = hist[hist["ISBN"].isin(D["row"])]
    liked = hist[hist["Book-Rating"] >= thr]
    if liked.empty:
        liked = hist.nlargest(min(5, len(hist)), "Book-Rating")
    return liked


def profile(isbns, weights=None):
    rows = [D["row"][i] for i in isbns if i in D["row"]]
    if not rows:
        return None
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, float)
    return sparse.csr_matrix(TFIDF[rows].multiply(w[:, None]).sum(axis=0) / max(w.sum(), 1e-9))


def because(rec_isbns, src_isbns):
    """Untuk tiap rekomendasi, cari buku sumber yang paling mirip (penjelasan 'karena kamu suka ...')."""
    src = [i for i in src_isbns if i in D["row"]]
    if not src:
        return [""] * len(rec_isbns)
    S = cosine_similarity(TFIDF[[D["row"][i] for i in rec_isbns]], TFIDF[[D["row"][i] for i in src]])
    out = []
    for k, j in enumerate(S.argmax(axis=1)):
        out.append(str(D["bidx"].loc[src[j], "Book-Title"]) if S[k, j] > 0.05 else "")
    return out


@st.cache_data(show_spinner="Menghitung rekomendasi hybrid...")
def recommend(uid, alpha, top_n, thr):
    hist = RATINGS[RATINGS["User-ID"] == int(uid)]
    cand = BOOKS[~BOOKS["ISBN"].isin(set(hist["ISBN"]))].copy()
    if hist.empty or cand.empty:
        return pd.DataFrame()
    isb = cand["ISBN"].tolist()
    est = np.array([CF.predict(int(uid), i).est for i in isb], float)
    cf_s = np.clip((est - 1) / 9, 0, 1)
    liked = liked_of(hist, thr)
    prof = profile(liked["ISBN"], liked["Book-Rating"].to_numpy(float) / 10)
    cbf = np.zeros(len(isb)) if prof is None else cosine_similarity(prof, TFIDF[[D["row"][i] for i in isb]]).ravel()
    cand["Prediksi CF"], cand["Skor CF"], cand["Skor CBF"] = est, cf_s, cbf
    cand["Skor Hybrid"] = alpha * cf_s + (1 - alpha) * cbf
    top = dedupe(cand.sort_values("Skor Hybrid", ascending=False), hist["ISBN"].tolist(), top_n).reset_index(drop=True)
    top["Karena"] = because(top["ISBN"].tolist(), liked["ISBN"].tolist())
    return top


@st.cache_data(show_spinner="Mencari buku yang mirip...")
def recommend_from_books(isbns, top_n):
    prof = profile(list(isbns))
    if prof is None:
        return pd.DataFrame()
    s = cosine_similarity(prof, TFIDF).ravel()
    cand = BOOKS.assign(**{"Skor CBF": s})
    cand = dedupe(cand[~cand["ISBN"].isin(isbns)].sort_values("Skor CBF", ascending=False), list(isbns), top_n).reset_index(drop=True)
    cand["Karena"] = because(cand["ISBN"].tolist(), list(isbns))
    return cand


def decade_of(y):
    return f"{int(y // 10 * 10)}-an" if pd.notna(y) and 1900 <= y <= 2006 else "Tidak diketahui"


@st.cache_resource(show_spinner=False)
def kproto():
    p = BASE / "kproto_final.joblib"
    return joblib.load(p) if p.exists() else None


def country_options():
    cu = D["cu"]
    if cu is None:
        return []
    return sorted(c for c in cu["Country_Cluster"].dropna().astype(str).unique() if c not in ("Other", "Unknown", "", "nan", "None"))


def predict_cluster(age, country):
    kp = kproto()
    if kp is None:
        raise FileNotFoundError("kproto_final.joblib belum ada")
    cval = country if country in country_options() else "Other"
    return int(kp.predict(np.array([[float(age), cval]], dtype=object), categorical=[1])[0])


def lookup_user(uid):
    """model = dikenal SVD (hybrid) · known = terdaftar di BX-Users tapi riwayat kurang · none = tidak ada."""
    if uid in D["ucount"]:
        return "model", None
    cu = D["cu"]
    if cu is not None:
        r = cu[cu["User-ID"] == uid]
        if len(r):
            return "known", r.iloc[0]
    return "none", None


def segment_books(cl, n):
    pb = D["pb"]
    cb = pb[pd.to_numeric(pb["Cluster"], errors="coerce") == cl].dropna(subset=["Book-Title"])
    cb = cb.sort_values(["Skor_Buku", "Jumlah_Rating"], ascending=False)
    return dedupe(cb, n=n) if n < 10**6 else cb


@st.cache_data(show_spinner="Menyusun rekomendasi pengguna baru...")
def newuser_recs(age, country, authors, publishers, decades, favs, n):
    """Skor = w·preferensi konten (TF-IDF penulis/penerbit/buku favorit) + (1-w)·popularitas segmen K-Prototypes."""
    cl = predict_cluster(age, country)
    pool = CAT[CAT["in_model"]].copy()
    pool = pool[~pool["ISBN"].isin(favs)]
    seg_map = segment_books(cl, 10**6).set_index("ISBN")["Skor_Buku"]
    seg = pool["ISBN"].map(seg_map).fillna(pool["bayes"]).fillna(pool["bayes"].min())
    pool["Skor segmen"] = (seg - seg.min()) / max(seg.max() - seg.min(), 1e-9)

    rows = TFIDF[[D["row"][i] for i in pool["ISBN"]]]
    pref = np.zeros(len(pool))
    words = " ".join(list(authors) + list(publishers))
    vec = D.get("vec")
    if words.strip() and vec is not None:
        pref = np.maximum(pref, cosine_similarity(vec.transform([words]), rows).ravel())
    elif words.strip():  # cadangan tanpa vectorizer: cocokkan teks langsung
        pref = np.maximum(pref, (pool["Book-Author"].isin(authors) | pool["Publisher"].isin(publishers)).astype(float).to_numpy())
    prof = profile(list(favs))
    if prof is not None:
        pref = np.maximum(pref, cosine_similarity(prof, rows).ravel())
    has_pref = pref.max() > 0
    pool["Skor preferensi"] = pref / pref.max() if has_pref else 0.0
    w = 0.7 if has_pref else 0.0
    pool["Skor akhir"] = w * pool["Skor preferensi"] + (1 - w) * pool["Skor segmen"]

    relaxed = False
    if decades:
        f = pool[pool["Year-Of-Publication"].map(decade_of).isin(decades)]
        relaxed = f.empty
        pool = pool if relaxed else f
    top = dedupe(pool.sort_values("Skor akhir", ascending=False), list(favs), n).reset_index(drop=True)

    why = because(top["ISBN"].tolist(), list(favs)) if favs else [""] * len(top)
    alasan = []
    for (_, r), k in zip(top.iterrows(), why):
        if r["Book-Author"] in authors:
            alasan.append(f"penulis favoritmu <b>{html.escape(str(r['Book-Author'])[:28])}</b>")
        elif str(r["Publisher"]) in publishers:
            alasan.append(f"penerbit favoritmu <b>{html.escape(str(r['Publisher'])[:28])}</b>")
        elif k and r["Skor preferensi"] > .3:
            alasan.append(f"mirip <b>{html.escape(k[:34])}</b>")
        elif r["ISBN"] in seg_map.index:
            alasan.append(f"favorit segmen <b>Cluster {cl}</b>")
        else:
            alasan.append("populer di semua pembaca")
    top["Alasan"] = alasan
    return cl, top, has_pref, relaxed


def search_books(df, q):
    k = q.strip()
    m = pd.Series(False, index=df.index)
    for c in ["Book-Title", "Book-Author", "Publisher", "ISBN"]:
        m |= df[c].astype(str).str.contains(k, case=False, regex=False, na=False)
    return df[m]


def similar(isbn, n=5):
    i = D["row"].get(isbn)
    if i is None:
        return pd.DataFrame()
    s = cosine_similarity(TFIDF[i], TFIDF).ravel()
    s[i] = -1
    return BOOKS.iloc[np.argsort(-s)[:n]]


def enrich(df):
    """Tambahkan statistik rating umum dan sampul ke tabel buku mana pun yang punya kolom ISBN."""
    df = df.copy()
    df["ISBN"] = df["ISBN"].astype(str)
    add = [c for c in ["n", "avg", "bayes", "Image-URL-M", "Image-URL-L", "Book-Title", "Book-Author"] if c not in df.columns]
    if add:
        df = df.merge(D["bidx"][add].reset_index(), on="ISBN", how="left")
    return df


# ---- Ringkasan EDA (dipakai prep_eda.py untuk data mentah & app.py untuk data model) ----
ALIAS = {"usa": "USA", "us": "USA", "u.s.a.": "USA", "united states": "USA", "uk": "United Kingdom", "u.k.": "United Kingdom",
         "england": "United Kingdom", "deutschland": "Germany", "espaa": "Spain", "españa": "Spain", "italia": "Italy"}


def summarize(B, R, U, m=20):
    """B = katalog (ISBN, judul, ...), R = interaksi (User-ID, ISBN, Book-Rating), U = profil (User-ID, Age, Country/Location)."""
    B = B.drop_duplicates("ISBN").copy()
    B["ISBN"] = B["ISBN"].astype(str)
    R = R.copy()
    R["ISBN"] = R["ISBN"].astype(str)
    R["Book-Rating"] = pd.to_numeric(R["Book-Rating"], errors="coerce")
    R["User-ID"] = pd.to_numeric(R["User-ID"], errors="coerce")
    valid = R["Book-Rating"].between(0, 10) & (R["Book-Rating"] % 1 == 0) & R["User-ID"].notna() & R["ISBN"].ne("")
    invalid = int((~valid).sum())
    R = R[valid]
    E = R[R["Book-Rating"] > 0]
    U = U.copy()
    U["User-ID"] = pd.to_numeric(U["User-ID"], errors="coerce")
    U = U.dropna(subset=["User-ID"]).drop_duplicates("User-ID")

    per_all = R.groupby("User-ID").size().reindex(U["User-ID"], fill_value=0)
    per_u, per_b = E.groupby("User-ID").size(), E.groupby("ISBN").size()
    act = pd.cut(per_all, [-1, 0, 1, 4, 19, 99, np.inf], labels=["0", "1", "2–4", "5–19", "20–99", "100+"]).value_counts(sort=False)
    bb = pd.cut(per_b, [0, 1, 4, 9, 19, 49, np.inf], labels=["1", "2–4", "5–9", "10–19", "20–49", "50+"]).value_counts(sort=False)

    arr = np.sort(per_all.to_numpy())[::-1]
    lorenz, top1 = [], 0.0
    if arr.sum():
        cum = np.r_[0, np.cumsum(arr)] / arr.sum() * 100
        idx = np.unique(np.r_[0, np.linspace(1, len(arr), min(200, len(arr)), dtype=int)])
        lorenz = [[round(i / len(arr) * 100, 3), round(float(cum[i]), 3)] for i in idx]
        top1 = float(arr[:max(1, int(np.ceil(len(arr) * .01)))].sum() / arr.sum() * 100)

    age = pd.to_numeric(U["Age"], errors="coerce") if "Age" in U else pd.Series(np.nan, index=U.index)
    va = age[age.between(5, 100)]
    ah = pd.cut(va, range(5, 105, 5), right=False).value_counts().sort_index()
    if "Country" in U:
        ctry = U["Country"].astype(str).str.strip()
    elif "Location" in U:
        ctry = U["Location"].fillna("").astype(str).str.rsplit(",", n=1).str[-1].str.strip()
    else:
        ctry = pd.Series("", index=U.index)
    low = ctry.str.lower().str.strip(' ."')
    ctry = low.map(ALIAS).fillna(ctry.str.title())
    ctry = ctry.where(~low.isin(["", "nan", "n/a", "none", "unknown", "other"]), "Tidak diketahui")
    cc = ctry.value_counts()
    known = cc.drop("Tidak diketahui", errors="ignore")
    countries = {k: int(v) for k, v in known.head(8).items()}
    if len(known) > 8:
        countries["Lainnya"] = int(known.iloc[8:].sum())

    yr = pd.to_numeric(B.get("Year-Of-Publication"), errors="coerce")
    vy = yr[yr.between(1950, 2004)]
    dec = (vy // 10 * 10).astype(int).value_counts().sort_index()
    EB = E.merge(B[["ISBN", "Book-Author", "Publisher"]], on="ISBN")
    au = EB.groupby("Book-Author").agg(readers=("User-ID", "nunique"), avg=("Book-Rating", "mean")).sort_values("readers", ascending=False).head(10)
    pub = EB.groupby("Publisher").agg(readers=("User-ID", "nunique")).sort_values("readers", ascending=False).head(10)

    ex_match = E[E["ISBN"].isin(B["ISBN"])]
    pairs = ex_match[["User-ID", "ISBN"]].drop_duplicates()
    possible = pairs["User-ID"].nunique() * pairs["ISBN"].nunique()
    inter = R.groupby("ISBN").size().rename("Interaksi")
    st_ = E.groupby("ISBN")["Book-Rating"].agg(["size", "mean"]).rename(columns={"size": "Rating eksplisit", "mean": "Rata-rata"})
    bm = B.merge(inter, on="ISBN", how="left").merge(st_, on="ISBN", how="left")
    for c in ("Interaksi", "Rating eksplisit"):
        bm[c] = bm[c].fillna(0).astype(int)
    C = float(ex_match["Book-Rating"].mean()) if len(ex_match) else 0.0
    n = bm["Rating eksplisit"]
    bm["Skor berbobot"] = ((n * bm["Rata-rata"].fillna(0) + m * C) / (n + m)).where(n > 0)
    unmatched = int((~R["ISBN"].isin(B["ISBN"])).sum())

    S = {
        "n_ratings": int(len(R)), "n_implicit": int((R["Book-Rating"] == 0).sum()), "n_explicit": int(len(E)),
        "n_users": int(len(U)), "n_books": int(len(B)), "n_users_explicit": int(per_u.size), "n_books_explicit": int(per_b.size),
        "mean_rating": float(E["Book-Rating"].mean()) if len(E) else 0.0, "catalog_mean": C,
        "median_user": float(per_u.median()) if len(per_u) else 0.0, "users_ge5": int((per_u >= 5).sum()),
        "books_ge10": int((per_b >= 10).sum()), "books_once": float((per_b == 1).mean()) if len(per_b) else 0.0,
        "isbn_match": float(1 - unmatched / max(len(R), 1)), "age_missing": float(age.isna().mean()),
        "age_out": int((age.notna() & ~age.between(5, 100)).sum()), "age_valid": int(len(va)),
        "age_median": float(va.median()) if len(va) else 0.0,
        "year_invalid": int((~yr.between(1950, 2004)).sum()), "density": float(len(pairs) / possible * 100) if possible else 0.0,
        "top1_share": top1, "lorenz": lorenz, "weight_m": m,
        "rating_dist": {str(k): int(E["Book-Rating"].astype(int).eq(k).sum()) for k in range(1, 11)},
        "activity": {str(k): int(v) for k, v in act.items()}, "book_bins": {str(k): int(v) for k, v in bb.items()},
        "age_hist": {f"{int(i.left)}": int(v) for i, v in ah.items()}, "countries": countries,
        "country_unknown": int(cc.get("Tidak diketahui", 0)),
        "authors": [{"author": k, "readers": int(r.readers), "avg": round(float(r.avg), 2)} for k, r in au.iterrows()],
        "publishers": [{"publisher": k, "readers": int(r.readers)} for k, r in pub.iterrows()],
        "decades": {str(k): int(v) for k, v in dec.items()},
        "coverage": {"Interaksi cocok dengan katalog": float(100 - unmatched / max(len(R), 1) * 100),
                     "ISBN katalog punya interaksi": float(bm["Interaksi"].gt(0).mean() * 100),
                     "ISBN katalog punya rating eksplisit": float(bm["Rating eksplisit"].gt(0).mean() * 100)},
        "audit": {"Baris rating tidak valid": invalid,
                  "Pasangan pengguna–ISBN berulang": int(R.duplicated(["User-ID", "ISBN"]).sum()),
                  "Interaksi tanpa metadata buku": unmatched,
                  "Interaksi tanpa profil pengguna": int((~R["User-ID"].isin(U["User-ID"])).sum()),
                  "ISBN tanpa rating eksplisit": int(bm["Rating eksplisit"].eq(0).sum()),
                  "Profil tanpa interaksi": int(per_all.eq(0).sum())},
    }
    cols = ["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication", "Interaksi", "Rating eksplisit", "Rata-rata", "Skor berbobot"]
    return S, bm[[c for c in cols if c in bm.columns]]


@st.cache_data(show_spinner="Menghitung ringkasan data model...")
def model_eda():
    U = pd.DataFrame({"User-ID": RATINGS["User-ID"].dropna().astype(int).unique()})
    if D["cu"] is not None:
        U = U.merge(D["cu"][["User-ID", "Age", "Country"]].drop_duplicates("User-ID"), on="User-ID", how="left")
    return summarize(BOOKS, RATINGS, U)



# =========================================================
# Komponen UI
# =========================================================
def https(u):
    return u.replace("http://", "https://", 1) if isinstance(u, str) else u


def cover(url, title="", cls="cover"):
    t = html.escape(str(title)[:50])
    ph = f'<div class="{cls} ph">{t}</div>'
    if isinstance(url, str) and url.startswith("http"):
        fallback = html.escape(ph).replace("'", "&#39;")
        return (f'<img class="{cls}" src="{html.escape(https(url))}" loading="lazy" alt="{t}" '
                f"onerror=\"this.outerHTML='{fallback}'\" "
                f"onload=\"if(this.naturalWidth<5)this.outerHTML='{fallback}'\">")
    return ph


def stars(avg10, n=None, big=False):
    """Bintang 1-5 dari rata-rata rating umum (skala 1-10 dibagi 2)."""
    if avg10 is None or pd.isna(avg10):
        return '<div class="strow"><span class="stars" style="--p:0%">★★★★★</span> belum ada rating</div>'
    s5 = float(avg10) / 2
    size = "font-size:1.5rem;" if big else ""
    cnt = f" · {short(n)} rating" if n is not None and pd.notna(n) else ""
    return (f'<div class="strow" title="Rata-rata rating umum {num(avg10, 2)}/10">'
            f'<span class="stars" style="--p:{s5 / 5 * 100:.1f}%;{size}">★★★★★</span>'
            f'<b>{num(s5, 1)}</b><span>{cnt}</span></div>')


def go_page(p):
    ss.page = p


def heading(title, kicker="", sub=""):
    st.markdown(f'<div class="sec"><div class="k">{kicker}</div><b>{title}</b><span>{sub}</span></div>', unsafe_allow_html=True)


def page_title(t, sub):
    st.markdown(f'<div class="page-title">{t}</div><div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def kpis(items):
    cols = st.columns(len(items))
    for c, (icon, grad, label, value, sub) in zip(cols, items):
        c.markdown(f'<div class="kpi"><div class="i {grad}">{icon}</div><div><div class="l">{label}</div>'
                   f'<div class="v">{value}</div><div class="s">{sub}</div></div></div>', unsafe_allow_html=True)


def note(text, green=False):
    st.markdown(f'<div class="note{" g" if green else ""}">{text}</div>', unsafe_allow_html=True)


@st.dialog("Detail buku", width="large")
def detail(isbn):
    if isbn not in D["bidx"].index:
        st.write("Data buku tidak ditemukan.")
        return
    info = D["bidx"].loc[isbn]
    a, b = st.columns([1, 2.2])
    a.markdown(f'<div class="bk"><div class="cv" style="max-width:190px">'
               f'{cover(info.get("Image-URL-L") or info.get("Image-URL-M"), info.get("Book-Title"))}</div></div>',
               unsafe_allow_html=True)
    yr = info.get("Year-Of-Publication")
    yr = int(yr) if pd.notna(yr) and yr > 0 else "-"
    b.markdown(f"### {html.escape(str(info.get('Book-Title', '')))}")
    b.markdown(f"**{info.get('Book-Author', '')}**  \n{info.get('Publisher', '')} · {yr} · ISBN {isbn}")
    b.markdown(stars(info.get("avg"), info.get("n"), big=True).replace('class="strow"', 'class="strow" style="justify-content:flex-start"'), unsafe_allow_html=True)
    if pd.notna(info.get("avg")):
        b.caption(f"Rata-rata rating umum {num(info['avg'], 2)}/10 dari {num(info['n'])} pembaca Book-Crossing "
                  f"(rating eksplisit 1-10, dikonversi ke 5 bintang).")
    sim = similar(isbn)
    if not sim.empty:
        st.markdown("**Buku dengan konten serupa** · berdasarkan kemiripan TF-IDF")
        sim = enrich(sim)
        cols = st.columns(len(sim))
        for c, (_, r) in zip(cols, sim.iterrows()):
            c.markdown(f'<div class="bk"><div class="cv" style="max-width:90px">'
                       f'{cover(r.get("Image-URL-M"), r["Book-Title"], "cover")}</div>'
                       f'<div class="bt" style="font-size:.76rem">{html.escape(str(r["Book-Title"])[:50])}</div>'
                       f'{stars(r.get("avg"))}</div>', unsafe_allow_html=True)


def grid(df, key, ncols=5, score=None, score_label="Kecocokan", rank=False, why=False):
    df = enrich(df).reset_index(drop=True)
    top = float(df[score].max()) if score and score in df and df[score].max() > 0 else 1.0
    for s in range(0, len(df), ncols):
        cols = st.columns(ncols)
        for c, (i, row) in zip(cols, df.iloc[s:s + ncols].iterrows()):
            isbn, t = str(row["ISBN"]), str(row.get("Book-Title", ""))
            badge = f'<div class="rank">{i + 1}</div>' if rank else ""
            extra = ""
            if score and pd.notna(row.get(score)):
                v = float(row[score])
                extra = (f'<div class="match">{score_label} {num(v, 3)}</div>'
                         f'<div class="mbar"><i style="width:{max(4, v / top * 100):.0f}%"></i></div>')
            if why and isinstance(row.get("Alasan"), str):
                extra += f'<div class="why">{row["Alasan"]}</div>'
            elif why:
                k = str(row.get("Karena") or "")
                extra += (f'<div class="why">mirip <b>{html.escape(k[:38])}</b></div>' if k
                          else '<div class="why">dari pola rating pembaca serupa</div>')
            with c, st.container(border=True):
                st.markdown(f'<div class="bk"><div class="cv">{badge}{cover(row.get("Image-URL-M"), t)}</div>'
                            f'<div class="bt">{html.escape(t)}</div>'
                            f'<div class="ba">{html.escape(str(row.get("Book-Author", "")))}</div>'
                            f'{stars(row.get("avg"), row.get("n"))}{extra}</div>', unsafe_allow_html=True)
                if st.button("Detail", key=f"{key}d{i}{isbn}", use_container_width=True):
                    detail(isbn)


def shelf(df, n=6):
    df = enrich(df).head(n)
    out = ""
    for _, r in df.iterrows():
        rt = f'<span class="tag">{int(r["Book-Rating"])}/10</span>' if "Book-Rating" in r else ""
        out += (f'<div class="row">{cover(r.get("Image-URL-M"), r.get("Book-Title"), "")}'
                f'<div><b>{html.escape(str(r["Book-Title"])[:60])}</b><small>{html.escape(str(r["Book-Author"])[:34])}</small> {rt}</div></div>')
    st.markdown(out, unsafe_allow_html=True)


def user_picker(key):
    fmt = lambda u: f"{u}  ·  {D['ucount'].get(u, 0)} rating"
    ss.user = st.selectbox("Pilih pembaca (User-ID)", D["uids"], index=D["uids"].index(ss.user), key=key, format_func=fmt,
                           help="Daftar diurutkan dari pembaca dengan riwayat rating terbanyak.")
    return ss.user


# ---------- Plotly ----------
def style(fig, h=330, legend=False):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=INK, size=12), bargap=.28,
        margin=dict(l=6, r=20, t=34 if legend else 10, b=6),
        legend=dict(orientation="h", y=1.12, x=0, title="", bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor=FOREST, bordercolor=FOREST, font=dict(color="#fff", family="Plus Jakarta Sans")),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#dcd3c0", title="", tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor="#ece5d6", zeroline=False, title="", tickfont=dict(color=MUTED, size=11))
    return fig


def chart_card(title, fig, sub="", insight=""):
    with st.container(border=True):
        st.markdown(f'<div class="ct">{title}</div>' + (f'<div class="cs">{sub}</div>' if sub else ""), unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if insight:
            st.markdown(f'<div class="ins">💡 {insight}</div>', unsafe_allow_html=True)


def bar(x, y, highlight=None, h=320, fmt=None, color=SAGE):
    cols = [GOLD if (highlight is not None and xi == highlight) else color for xi in x]
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker=dict(color=cols, cornerradius=8, line=dict(width=0)),
                           text=[fmt(v) for v in y] if fmt else None, textposition="outside", cliponaxis=False,
                           textfont=dict(size=10, color=MUTED), hovertemplate="%{x}<br><b>%{y:,}</b><extra></extra>"))
    return style(fig, h)


def hbar(labels, values, h=None, fmt=lambda v: f"{v:,}"):
    labels, values = list(labels)[::-1], list(values)[::-1]
    cols = [SAGE] * (len(values) - 1) + [GOLD]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=cols, cornerradius=7),
                           text=[fmt(v) for v in values], textposition="outside", cliponaxis=False,
                           textfont=dict(size=10, color=MUTED), hovertemplate="%{y}<br><b>%{x:,}</b><extra></extra>"))
    fig = style(fig, h or max(300, 32 * len(values)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True, tickfont=dict(color=INK, size=11))
    return fig


def donut(labels, values, colors, center, h=280):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.68, marker=dict(colors=colors, line=dict(color="#fff", width=3)),
                           textinfo="percent", textfont=dict(size=12, color="#fff"), sort=False,
                           hovertemplate="%{label}<br><b>%{value:,}</b> (%{percent})<extra></extra>"))
    fig.add_annotation(text=center, showarrow=False, font=dict(family="Fraunces, serif", size=18, color=FOREST))
    fig = style(fig, h, legend=True)
    fig.update_layout(legend=dict(orientation="h", y=-0.05, x=.5, xanchor="center"))
    return fig


# =========================================================
# Sidebar
# =========================================================
PAGES = [("Beranda", "🏠"), ("Rekomendasi", "✨"), ("Katalog", "📚"), ("Eksplorasi Data", "📊"), ("Evaluasi & Segmentasi", "🎯")]
with st.sidebar:
    st.markdown('<div class="logo"><div class="ic">📖</div><div><b>Book<br>Recommender</b><small>DASHBOARD</small></div></div>',
                unsafe_allow_html=True)
    for p, ic in PAGES:
        st.button(f"{ic}   {p}", key=f"nav{p}", on_click=go_page, args=(p,), use_container_width=True,
                  type="primary" if ss.page == p else "secondary")
    st.markdown(f'<div class="side-note"><b>Mesin utama</b><br>Hybrid = {BEST_ALPHA:.1f}·CF (SVD) + {1 - BEST_ALPHA:.1f}·CBF (TF-IDF).'
                f'<br><br><b>Pengguna baru</b><br>Form usia &amp; negara (K-Prototypes) + preferensi penulis, penerbit, dekade.'
                f'<br><br><b>★ Bintang</b><br>Rating umum 1-10 seluruh pembaca, dikonversi ke 5 bintang.</div>',
                unsafe_allow_html=True)

page = ss.page
RAW = D["eda"]
E = RAW

# =========================================================
# BERANDA
# =========================================================
if page == "Beranda":
    with st.container(key="hero"):
        st.markdown(
            '<div class="pill">📖 Book-Crossing · Sistem Rekomendasi Hybrid</div>'
            '<div class="hl">Discover your next<br><em>favorite book</em></div>'
            '<p class="sub">Rekomendasi personal dari gabungan pola rating pembaca (Collaborative Filtering) '
            'dan kemiripan isi buku (Content-Based Filtering).</p>'
            f'<div class="chips"><span class="chip"><b>{num(len(BOOKS))}</b>buku kandidat</span>'
            f'<span class="chip"><b>{num(RATINGS["User-ID"].nunique())}</b>pembaca model</span>'
            f'<span class="chip"><b>{num(len(RATINGS))}</b>rating latih</span></div>', unsafe_allow_html=True)
        st.button("Mulai rekomendasi  →", key="cta", on_click=go_page, args=("Rekomendasi",))

    heading("Cari buku", "Pencarian", "judul, penulis, penerbit, atau ISBN")
    hq = st.text_input("Cari buku", placeholder="🔎  mis. Harry Potter, Grisham, Penguin, 0439...", key="home_q",
                       label_visibility="collapsed")
    if hq.strip():
        hits = search_books(CAT, hq).sort_values("n", ascending=False)
        if hits.empty:
            st.warning("Tidak ada buku yang cocok. Coba kata kunci lain.")
        else:
            st.caption(f"{num(len(hits))} buku ditemukan · menampilkan 5 teratas")
            grid(hits.head(5), "hs")
            st.button("Lihat semua hasil di Katalog →", key="to_cat",
                      on_click=lambda: ss.update(page="Katalog", cat_q=hq))

    heading("Bagaimana rekomendasi dibuat", "Cara kerja", "satu mesin utama, dua jalur untuk pengguna baru")
    s = st.columns(4)
    steps = [("#1f6b4f", "01 · CF", "Collaborative Filtering", "SVD mempelajari pola rating ribuan pembaca untuk memprediksi rating buku yang belum dibaca."),
             ("#e3a83b", "02 · CBF", "Content-Based", "TF-IDF judul, penulis, penerbit, tahun. Buku yang mirip dengan favoritmu mendapat skor tinggi."),
             ("#0f3d2e", "03 · HYBRID", "Skor gabungan", f"Skor = α·CF + (1−α)·CBF, dengan α = {BEST_ALPHA:.1f} hasil evaluasi NDCG@10."),
             ("#8c2f39", "04 · BARU", "Pengguna baru", "Isi form usia, negara, serta penulis, penerbit, atau buku favorit: K-Prototypes + kemiripan konten.")]
    for c, (col, n, t, d) in zip(s, steps):
        c.markdown(f'<div class="step" style="--c:{col}"><div class="n">{n}</div><b>{t}</b><p>{d}</p></div>', unsafe_allow_html=True)

    heading("Rekomendasi untukmu", "Personal", "skor hybrid untuk pembaca terpilih")
    c1, c2 = st.columns([1.1, 3])
    with c1:
        u = user_picker("home_user")
        hist = RATINGS[RATINGS["User-ID"] == u]
        st.markdown(f'<div class="note g">Pembaca <b>{u}</b> sudah menilai <b>{len(hist)}</b> buku dengan rata-rata '
                    f'<b>{num(hist["Book-Rating"].mean(), 1)}/10</b>.</div>', unsafe_allow_html=True)
        st.button("Lihat semua rekomendasi →", on_click=go_page, args=("Rekomendasi",), use_container_width=True, key="w_button_lihat_semua_rekomendasi")
    with c2:
        recs = recommend(u, BEST_ALPHA, 4, LIKE_THR)
        if recs.empty:
            st.info("Belum ada rekomendasi untuk pembaca ini.")
        else:
            grid(recs, "home", ncols=4, score="Skor Hybrid", rank=True)

    heading("Buku populer", "Trending", "dari seluruh rating eksplisit Book-Crossing")
    a, b = st.columns([3, 1])
    mode = a.radio("Urutkan", ["🔥 Paling banyak dinilai", "⭐ Rating tertinggi"], horizontal=True, label_visibility="collapsed", key="w_radio_urutkan")
    n_show = b.select_slider("Jumlah", [5, 10, 15], value=5, label_visibility="collapsed", key="w_select_slider_jumlah")
    pop = CAT[CAT["in_model"]]
    if mode.startswith("🔥"):
        pop = pop.sort_values("n", ascending=False)
    else:
        pop = pop[pop["n"] >= 30].sort_values("bayes", ascending=False)
        st.caption("Diurutkan dengan Bayesian Weighted Rating (minimal 30 rating) agar buku dengan sedikit rating tidak mendominasi.")
    grid(dedupe(pop, n=n_show), "pop", rank=True)

    E = RAW or model_eda()[0]
    if E:
        heading("Penulis teratas", "Top authors", "berdasarkan jumlah pembaca unik yang memberi rating")
        au = E["authors"][:6]
        for rank, (c, r) in enumerate(zip(st.columns(6), au), 1):
            ini = "".join(w[0] for w in str(r["author"]).replace(".", " ").split()[:2]).upper()
            with c, st.container(border=True):
                st.markdown(f'<div class="au"><div class="rk">#{rank}</div><div class="av">{html.escape(ini)}</div>'
                            f'<div class="nm">{html.escape(r["author"])}</div><div class="sm">{num(r["readers"])} pembaca</div>'
                            f'{stars(r["avg"])}</div>', unsafe_allow_html=True)
                st.button("Lihat buku", key=f"au{rank}", use_container_width=True,
                          on_click=lambda x=r["author"]: ss.update(author=None if ss.author == x else x))
        if ss.author:
            heading(f"Buku karya {ss.author}", "Penulis", "klik tombolnya lagi untuk menutup")
            ab = CAT[CAT["Book-Author"].str.lower() == ss.author.lower()].sort_values("n", ascending=False).head(10)
            grid(ab, "aub") if len(ab) else st.info("Buku penulis ini tidak ada di katalog kandidat.")

# =========================================================
# REKOMENDASI
# =========================================================
elif page == "Rekomendasi":
    page_title("Rekomendasi buku", "Masukkan User-ID untuk rekomendasi personal, atau isi form jika kamu pengguna baru.")
    t1, t2 = st.tabs(["🔎  Masukkan User-ID", "📝  Form pengguna baru"])

    # ---- Ketik User-ID ----
    with t1:
        ss.setdefault("uid_in", int(ss.user))

        def _pick():
            ss.uid_in = int(ss.quick_uid)

        c1, c2, c3, c4, c5 = st.columns([1.1, 1.3, 1, 1, 1])
        uid = int(c1.number_input("Ketik User-ID", min_value=1, step=1, key="uid_in",
                                  help="User-ID dari dataset Book-Crossing (BX-Users)."))
        c2.selectbox("…atau pilih pembaca aktif", D["uids"], key="quick_uid", on_change=_pick,
                     index=D["uids"].index(ss.user) if ss.user in D["uids"] else 0,
                     format_func=lambda u: f"{u}  ·  {D['ucount'].get(u, 0)} rating")
        top_n = c3.select_slider("Jumlah buku", [5, 10, 15, 20], value=10, key="w_select_slider_jumlah_buku")
        alpha = c4.slider("α (bobot CF)", 0.0, 1.0, float(np.clip(BEST_ALPHA, 0, 1)), 0.05,
                          help=f"0 = murni konten, 1 = murni CF. Nilai terbaik hasil evaluasi: {BEST_ALPHA}.", key="w_slider_bobot_cf")
        thr = c5.slider("Ambang 'disukai'", 1, 10, LIKE_THR, help="Rating minimal agar buku dianggap disukai (membentuk profil CBF).", key="w_slider_ambang_disukai")

        status, urow = lookup_user(uid)
        if status == "none":
            st.warning(f"User-ID **{uid}** tidak ditemukan di data Book-Crossing. Gunakan tab **Form pengguna baru**, "
                       "atau pilih salah satu pembaca aktif.")
        elif status == "known":
            age_txt = f"{urow['Age']:.0f} tahun" if pd.notna(urow.get("Age")) else "usia tidak diisi"
            cl = int(urow["Cluster"])
            note(f"User-ID <b>{uid}</b> terdaftar ({age_txt}, {html.escape(str(urow.get('Country', '-')))}), tetapi riwayat ratingnya "
                 f"kurang dari {META.get('min_user_ratings', 20)} buku sehingga <b>belum dikenal model hybrid</b>. "
                 f"Rekomendasi diambil dari segmen K-Prototypes-nya: <b>Cluster {cl}</b>. Isi tab Form pengguna baru "
                 "untuk hasil yang lebih personal.")
            cb = segment_books(cl, top_n) if D["pb"] is not None else pd.DataFrame()
            heading(f"Rekomendasi untuk pembaca {uid}", "Segmen K-Prototypes", "skor segmen = Bayesian Weighted Rating dalam cluster")
            if cb.empty:
                st.info("Belum ada buku untuk segmen ini.")
            else:
                grid(cb, "known", score="Skor_Buku", score_label="Skor segmen", rank=True)
        else:
            ss.user = uid
            hist = RATINGS[RATINGS["User-ID"] == uid].sort_values("Book-Rating", ascending=False)
            liked = liked_of(hist, thr)
            top_au = enrich(hist)["Book-Author"].value_counts()
            kpis([("📚", "g1", "Buku dinilai", num(len(hist)), "riwayat di data latih"),
                  ("⭐", "g2", "Rata-rata rating", f"{num(hist['Book-Rating'].mean(), 1)}/10", f"≈ {num(hist['Book-Rating'].mean() / 2, 1)} bintang"),
                  ("❤️", "g3", "Buku disukai", num(len(liked)), f"rating ≥ {thr} → profil konten"),
                  ("✍️", "g4", "Penulis favorit", html.escape(str(top_au.index[0])[:18]) if len(top_au) else "-",
                   f"{top_au.iloc[0]} buku dinilai" if len(top_au) else "")])

            recs = recommend(uid, alpha, top_n, thr)
            left, right = st.columns([3.3, 1])
            with right:
                heading("Rak favoritnya", "Profil", "buku dengan rating tertinggi")
                with st.container(border=True):
                    shelf(hist, 6)
            with left:
                heading(f"Top {top_n} untuk pembaca {uid}", "Hybrid CF + CBF",
                        "bar hijau-emas = skor hybrid relatif terhadap peringkat #1")
                if recs.empty:
                    st.warning("Tidak ada buku baru untuk pembaca ini.")
                else:
                    grid(recs, "hy", score="Skor Hybrid", score_label="Skor hybrid", rank=True, why=True)

            if not recs.empty:
                lab = [f"{i + 1}. {str(t)[:34]}" for i, t in enumerate(recs["Book-Title"])]
                fig = go.Figure()
                fig.add_bar(y=lab, x=alpha * recs["Skor CF"], name=f"Kontribusi CF (α={alpha:.2f})", orientation="h",
                            marker=dict(color=FOREST, cornerradius=4))
                fig.add_bar(y=lab, x=(1 - alpha) * recs["Skor CBF"], name=f"Kontribusi CBF (1−α={1 - alpha:.2f})", orientation="h",
                            marker=dict(color=GOLD, cornerradius=4))
                fig.update_layout(barmode="stack")
                fig = style(fig, h=max(320, 30 * len(recs)), legend=True)
                fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK))
                a, b = st.columns([1.6, 1])
                with a:
                    chart_card("Dari mana skor tiap rekomendasi berasal", fig, "skor hybrid = α·CF + (1−α)·CBF",
                               "Bar emas yang panjang berarti buku itu direkomendasikan karena <b>mirip</b> dengan buku yang disukai; "
                               "bar hijau berarti didorong oleh <b>pola rating</b> pembaca lain.")
                with b:
                    f2 = go.Figure(go.Scatter(x=recs["Skor CBF"], y=recs["Prediksi CF"], mode="markers+text",
                                              text=[str(i + 1) for i in range(len(recs))], textposition="middle center",
                                              textfont=dict(color="#fff", size=10),
                                              marker=dict(size=24, color=recs["Skor Hybrid"], colorscale=[[0, SAGE], [1, GOLD]],
                                                          line=dict(color=FOREST, width=1.5)),
                                              customdata=recs["Book-Title"],
                                              hovertemplate="%{customdata}<br>CBF %{x:.3f} · prediksi CF %{y:.2f}/10<extra></extra>"))
                    f2 = style(f2, h=max(320, 30 * len(recs)))
                    f2.update_xaxes(title="Kemiripan konten (CBF)", showgrid=True, gridcolor="#ece5d6")
                    f2.update_yaxes(title="Prediksi rating SVD (1-10)")
                    chart_card("Peta rekomendasi", f2, "nomor = peringkat",
                               "Kanan atas = cocok dari dua sisi: mirip konten dan diprediksi dinilai tinggi.")
                st.download_button("⬇️  Unduh rekomendasi (CSV)",
                                   recs[["ISBN", "Book-Title", "Book-Author", "Prediksi CF", "Skor CF", "Skor CBF", "Skor Hybrid"]]
                                   .to_csv(index=False).encode("utf-8-sig"), f"rekomendasi_user_{uid}.csv", "text/csv", key="w_download_button_unduh_rekomendasi_csv")

    # ---- Form pengguna baru ----
    with t2:
        note("Belum punya User-ID? Isi <b>usia dan negara</b> (wajib) agar K-Prototypes menempatkanmu ke segmen pembaca yang mirip. "
             "Tambahkan <b>penulis, penerbit, dekade, atau buku favorit</b> (opsional) supaya hasilnya lebih personal. "
             "Model SVD tidak dipakai karena belum mengenal pengguna baru.", green=True)
        pool = CAT[CAT["in_model"]]
        ctry = country_options()
        au_opts = pool["Book-Author"].value_counts().index.tolist()
        pub_opts = pool["Publisher"].astype(str).value_counts().index.tolist()
        dec_opts = sorted(d for d in pool["Year-Of-Publication"].map(decade_of).unique() if d != "Tidak diketahui")
        pop_books = pool.sort_values("n", ascending=False)
        blabel = dict(zip(pop_books["ISBN"], pop_books["Book-Title"].str.slice(0, 60) + " — " + pop_books["Book-Author"].astype(str)))
        with st.form("form_baru", border=True):
            st.markdown('<div class="ct">📝 Profil pembaca baru</div>', unsafe_allow_html=True)
            a, b, c = st.columns([1.4, .8, 1.2])
            nama = a.text_input("Nama (opsional)", placeholder="mis. Andi", key="w_text_input_nama_opsional")
            age = b.number_input("Usia *", 5, 100, 25, key="w_number_input_usia")
            country = c.selectbox("Negara *", ctry + ["Lainnya"], index=ctry.index("USA") if "USA" in ctry else 0, key="w_selectbox_negara")
            a, b, c = st.columns(3)
            authors = a.multiselect("Penulis favorit", au_opts, max_selections=5, placeholder="Ketik nama penulis...", key="w_multiselect_penulis_favorit")
            pubs = b.multiselect("Penerbit favorit", pub_opts, max_selections=5, placeholder="Ketik nama penerbit...", key="w_multiselect_penerbit_favorit")
            decs = c.multiselect("Dekade terbit", dec_opts, placeholder="Semua dekade", key="w_multiselect_dekade_terbit")
            a, b = st.columns([3, 1])
            favs = a.multiselect("Buku yang pernah kamu sukai", list(blabel), format_func=blabel.get, max_selections=5,
                                 placeholder="Ketik judul buku...", key="w_multiselect_buku_yang_pernah_kamu_sukai")
            n_new = b.select_slider("Jumlah rekomendasi", [5, 10, 15, 20], value=10, key="w_select_slider_jumlah_rekomendasi")
            sent = st.form_submit_button("✨  Tampilkan rekomendasi", type="primary", use_container_width=True, key="w_form_submit_button_tampilkan_rekomendasi")
        if sent:
            ss.nu = dict(nama=nama.strip(), age=int(age), country=country, authors=tuple(authors), pubs=tuple(pubs),
                         decs=tuple(decs), favs=tuple(favs), n=n_new)
        nu = ss.get("nu")
        if not nu:
            st.caption("Isi form lalu klik **Tampilkan rekomendasi**.")
        else:
            try:
                cl, top, has_pref, relaxed = newuser_recs(nu["age"], nu["country"], nu["authors"], nu["pubs"],
                                                          nu["decs"], nu["favs"], nu["n"])
            except Exception as exc:
                st.error(f"Gagal memuat model clustering: {exc}")
                st.caption("Periksa versi kmodes dan scikit-learn, serta urutan fitur (Age, Country_Cluster).")
            else:
                cu = D["cu"]
                prof = cu[pd.to_numeric(cu["Cluster"], errors="coerce") == cl] if cu is not None else pd.DataFrame()
                n_pref = len(nu["authors"]) + len(nu["pubs"]) + len(nu["favs"])
                kpis([("🧭", "g1", "Segmen kamu", f"Cluster {cl}", f"usia {nu['age']} · {nu['country']}"),
                      ("👥", "g4", "Anggota segmen", short(len(prof)), f"usia rata-rata {pd.to_numeric(prof['Age'], errors='coerce').mean():.0f} th" if len(prof) else ""),
                      ("🎯", "g2", "Preferensi diisi", num(n_pref), "penulis · penerbit · buku"),
                      ("🗓️", "g3", "Dekade", ", ".join(nu["decs"]) if nu["decs"] else "Semua", "filter tahun terbit")])
                if relaxed:
                    st.warning("Tidak ada buku pada dekade yang dipilih, jadi filter dekade diabaikan.")
                sapa = f"Untuk {html.escape(nu['nama'])}" if nu["nama"] else "Rekomendasi untukmu"
                sub = ("skor = 0,7 × kecocokan preferensi + 0,3 × popularitas segmen" if has_pref
                       else "tanpa preferensi: diurutkan dari popularitas segmen (Bayesian Weighted Rating)")
                heading(sapa, f"Pengguna baru · Cluster {cl}", sub)
                grid(top, "nu", score="Skor akhir", score_label="Skor", rank=True, why=True)

# =========================================================
# KATALOG
# =========================================================
elif page == "Katalog":
    page_title("Katalog buku", "Cari berdasarkan judul, penulis, penerbit, atau ISBN, lalu saring dengan dekade dan bintang.")
    base = CAT.copy()
    base["Dekade"] = base["Year-Of-Publication"].map(decade_of)
    ss.setdefault("cat_q", "")
    with st.container(border=True):
        a, b, c = st.columns([2.4, 1.3, 1.3])
        q = a.text_input("Cari", placeholder="🔎  Judul, penulis, penerbit, atau ISBN...", key="cat_q", label_visibility="collapsed")
        au = b.selectbox("Penulis", ["Semua penulis"] + base["Book-Author"].value_counts().index.tolist(), label_visibility="collapsed", key="w_selectbox_penulis")
        pub = c.selectbox("Penerbit", ["Semua penerbit"] + base["Publisher"].astype(str).value_counts().index.tolist(),
                          label_visibility="collapsed", key="w_selectbox_penerbit")
        a, b, c = st.columns([2.4, 1.3, 1.3])
        decs = a.multiselect("Dekade", sorted(base["Dekade"].unique()), placeholder="Semua dekade", label_visibility="collapsed", key="w_multiselect_dekade")
        min_s = b.select_slider("Bintang minimal", [0, 3, 3.5, 4, 4.5], value=0,
                                format_func=lambda v: "Semua ★" if v == 0 else f"≥ {v} ★", label_visibility="collapsed", key="w_select_slider_bintang_minimal")
        sort = c.selectbox("Urutkan", ["Terpopuler", "Bintang tertinggi", "Terbaru", "Judul A-Z"], label_visibility="collapsed", key="w_selectbox_urutkan")
    res = search_books(base, q) if q.strip() else base
    if au != "Semua penulis":
        res = res[res["Book-Author"] == au]
    if pub != "Semua penerbit":
        res = res[res["Publisher"].astype(str) == pub]
    if decs:
        res = res[res["Dekade"].isin(decs)]
    if min_s:
        res = res[res["avg"] / 2 >= min_s]
    res = {"Terpopuler": res.sort_values("n", ascending=False),
           "Bintang tertinggi": res.sort_values("bayes", ascending=False),
           "Terbaru": res.sort_values("Year-Of-Publication", ascending=False),
           "Judul A-Z": res.sort_values("Book-Title")}[sort]
    per = 18
    pages = max(1, int(np.ceil(len(res) / per)))
    h1, h2 = st.columns([3, 1])
    h1.markdown(f'<div class="sec" style="margin-top:.8rem"><span><b style="display:inline;font-size:1.3rem">{num(len(res))}</b> buku ditemukan</span></div>',
                unsafe_allow_html=True)
    pg = h2.number_input("Halaman", 1, pages, 1, help=f"dari {pages} halaman", key=f"pg_{len(res)}_{sort}") if pages > 1 else 1
    if res.empty:
        st.warning("Tidak ada buku yang cocok. Coba kata kunci lain atau longgarkan filter.")
    else:
        grid(res.iloc[(pg - 1) * per: pg * per], f"cat{pg}", ncols=6)

# =========================================================
# EKSPLORASI DATA
# =========================================================
elif page == "Eksplorasi Data":
    page_title("Eksplorasi data", "Buku, pembaca, dan seberapa kuat bukti yang dipakai sistem rekomendasi.")

    scopes = (["🌐  Seluruh dataset Book-Crossing"] if RAW else []) + ["🧠  Data yang dipakai model"]
    scope = st.radio("Cakupan", scopes, horizontal=True, key="eda_scope", label_visibility="collapsed")
    if scope.startswith("🌐"):
        E, BM = RAW, D["eda_books"]
        st.caption("Ringkasan seluruh data mentah (1,15 jt interaksi), dihitung sekali dengan prep_eda.py. "
                   "Tabel buku memuat ISBN dengan ≥ 10 rating eksplisit.")
    else:
        E, BM = model_eda()
        if not RAW:
            st.info("Tambahkan eda_summary.json dan eda_books.csv (hasil prep_eda.py) untuk melihat seluruh dataset mentah.")
        st.caption(f"Hanya data latih model: pembaca dengan ≥ {META.get('min_user_ratings', 20)} rating dan buku dengan "
                   f"≥ {META.get('min_book_ratings', 10)} rating. Rating 0 sudah dibuang, jadi semua interaksi di sini eksplisit.")
    raw = scope.startswith("🌐")

    kpis([("📚", "g1", "ISBN dalam katalog", short(E["n_books"]), f"{short(E['n_books_explicit'])} punya rating eksplisit"),
          ("👥", "g4", "Profil pembaca", short(E["n_users"]), f"{short(E['n_users_explicit'])} memberi rating eksplisit"),
          ("🗂️", "g3", "Interaksi valid", short(E["n_ratings"]), f"{E['n_explicit'] / max(E['n_ratings'], 1) * 100:.0f}% eksplisit"),
          ("⭐", "g2", "Rata-rata rating", f"{num(E['mean_rating'], 2)}/10", f"≈ {num(E['mean_rating'] / 2, 1)} bintang")])

    t1, t2, t3, t4 = st.tabs(["📈  Ringkasan & aktivitas", "🔎  Jelajahi buku", "👥  Profil pembaca", "🧪  Kualitas & cakupan"])

    # ---------- Ringkasan & aktivitas ----------
    with t1:
        rd = E["rating_dist"]
        a, b = st.columns([1, 1.5])
        with a:
            if E["n_implicit"]:
                exp_share = E["n_explicit"] / E["n_ratings"]
                chart_card("Komposisi umpan balik",
                           donut(["Implisit (rating 0)", "Eksplisit (1-10)"], [E["n_implicit"], E["n_explicit"]], [SAGE, FOREST],
                                 f"{exp_share * 100:.0f}%<br><span style='font-size:11px'>eksplisit</span>"),
                           "rating 0 = interaksi tanpa nilai",
                           f"<b>{(1 - exp_share) * 100:.0f}%</b> interaksi bernilai 0. Nilai 0 <b>bukan skor buruk</b>, jadi tidak "
                           "dihitung ke rata-rata dan dibuang dari pelatihan SVD.")
            else:
                s5 = {str(i): 0 for i in range(1, 6)}
                for k, v in rd.items():
                    s5[str(int(np.ceil(int(k) / 2)))] += v
                tot = max(sum(s5.values()), 1)
                rows = "".join(
                    f'<div style="display:flex;align-items:center;gap:.7rem;margin:.45rem 0"><span style="width:62px;color:#e3a83b">{"★" * int(k)}</span>'
                    f'<div style="flex:1;height:12px;background:#efe8d8;border-radius:8px;overflow:hidden"><div style="width:{s5[k] / tot * 100:.1f}%;'
                    f'height:100%;background:linear-gradient(90deg,#1f6b4f,#e3a83b);border-radius:8px"></div></div>'
                    f'<span style="width:100px;text-align:right;font-size:.8rem;color:#6f7d74">{s5[k] / tot * 100:.1f}%</span></div>'
                    for k in ["5", "4", "3", "2", "1"])
                with st.container(border=True):
                    st.markdown('<div class="ct">Rating dalam 5 bintang</div><div class="cs">semua interaksi data model eksplisit</div>'
                                + rows + '<div class="ins">💡 9-10 = 5★, 7-8 = 4★, 5-6 = 3★, 3-4 = 2★, 1-2 = 1★.</div>',
                                unsafe_allow_html=True)
        with b:
            mx = max(rd, key=rd.get)
            hi = sum(v for k, v in rd.items() if int(k) >= 7) / max(sum(rd.values()), 1) * 100
            chart_card("Sebaran skor eksplisit", bar(list(rd), list(rd.values()), highlight=mx, h=300, fmt=short),
                       "jumlah rating per nilai 1-10",
                       f"Skor <b>{mx}</b> paling sering muncul dan {hi:.0f}% rating bernilai ≥ 7. Pembaca cenderung hanya menilai "
                       "buku yang mereka sukai (<b>positivity bias</b>).")
        a, b = st.columns(2)
        with a:
            act = E["activity"]
            chart_card("Interaksi per profil pembaca", bar(list(act), list(act.values()), highlight=max(act, key=act.get), fmt=short),
                       "termasuk rating 0 · kelompok 0 = tidak ada interaksi",
                       f"<b>{num(act.get('0', 0))}</b> profil ({act.get('0', 0) / max(E['n_users'], 1) * 100:.0f}%) tidak punya interaksi, "
                       f"dan median rating eksplisit per pembaca hanya <b>{E['median_user']:.0f}</b>. Riwayat pendek menyulitkan CF.")
        with b:
            lz = np.array(E["lorenz"]) if E["lorenz"] else np.zeros((0, 2))
            fig = go.Figure()
            fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Merata sempurna", line=dict(color=SAGE, dash="dash", width=2))
            fig.add_scatter(x=lz[:, 0], y=lz[:, 1], mode="lines", name="Aktivitas sebenarnya", fill="tozeroy",
                            fillcolor="rgba(227,168,59,.14)", line=dict(color=FOREST, width=3),
                            hovertemplate="%{x:.1f}% profil teraktif<br>menyumbang <b>%{y:.1f}%</b> interaksi<extra></extra>")
            fig = style(fig, 320, legend=True)
            fig.update_xaxes(ticksuffix="%", title="profil, dari yang paling aktif")
            fig.update_yaxes(ticksuffix="%", title="kumulatif interaksi")
            chart_card("Konsentrasi aktivitas", fig, "kurva Lorenz",
                       f"Hanya <b>1%</b> profil teraktif menyumbang <b>{E['top1_share']:.0f}%</b> interaksi. Model mudah didominasi "
                       "selera segelintir pembaca berat.")
        a, b = st.columns([1, 1.3] if raw else [1, 1])
        with a:
            bb = E["book_bins"]
            chart_card("Rating eksplisit per buku", bar(list(bb), list(bb.values()), highlight=max(bb, key=bb.get), fmt=short, color="#c9b48a"),
                       "long tail katalog",
                       f"<b>{E['books_once'] * 100:.0f}%</b> buku hanya dinilai sekali; hanya {num(E['books_ge10'])} buku punya ≥ 10 rating. "
                       "Untuk buku seperti ini, CBF menjadi pelengkap CF.")
        if raw:
            with b:
                fig = go.Figure(go.Funnel(y=["Semua interaksi", "Rating eksplisit", "Data model"],
                                          x=[E["n_ratings"], E["n_explicit"], META.get("n_model_ratings", len(RATINGS))],
                                          textinfo="value+percent initial", marker=dict(color=[SAGE, PINE, GOLD]),
                                          connector=dict(fillcolor="#efe8d8")))
                fig = style(fig, 320)
                fig.update_yaxes(tickfont=dict(color=INK, size=12))
                chart_card("Dari data mentah ke data latih", fig, f"filter: pembaca ≥ {META.get('min_user_ratings', 20)} rating, buku ≥ {META.get('min_book_ratings', 10)} rating",
                           f"Tersisa <b>{num(META.get('n_model_users', 0))}</b> pembaca dan <b>{num(META.get('n_model_books', len(BOOKS)))}</b> buku "
                           "dengan sinyal yang cukup untuk dipelajari.")
        else:
            with b:
                with st.container(border=True):
                    st.markdown(f'<div class="ct">Kepadatan matriks</div><div class="cs">pasangan pembaca–buku yang punya rating</div>'
                                f'<div class="kpi" style="box-shadow:none;border:0;padding:.4rem 0"><div class="i g2">🧩</div><div>'
                                f'<div class="v">{E["density"]:.3f}%</div><div class="s">di data model, dibandingkan '
                                f'{RAW["density"]:.4f}% di data mentah</div></div></div>' if RAW else
                                f'<div class="ct">Kepadatan matriks</div><div class="v">{E["density"]:.3f}%</div>', unsafe_allow_html=True)
                    st.markdown('<div class="ins">💡 Penyaringan membuat matriks jauh lebih padat sehingga SVD bisa mempelajari pola, '
                                'tetapi buku dan pembaca di luar filter hanya bisa dilayani CBF atau cluster.</div>', unsafe_allow_html=True)

    # ---------- Jelajahi buku ----------
    with t2:
        with st.container(border=True):
            f1, f2, f3 = st.columns([2.2, 1, 1])
            query = f1.text_input("Cari judul, penulis, atau penerbit", key="eda_search", placeholder="🔎  mis. Grisham, Penguin...")
            mins = [10, 30, 50, 100] if raw else [0, 10, 30, 50, 100]
            threshold = f2.selectbox("Minimal rating eksplisit", mins, index=1 if raw else 2, key="eda_min")
            top_n = f3.selectbox("Jumlah buku di grafik", [5, 10, 15], index=1, key="eda_top")
            order = st.radio("Urutkan berdasarkan", ["Skor berbobot", "Rating eksplisit", "Interaksi", "Rata-rata"],
                             horizontal=True, key="eda_order")
        view = BM[BM["Rating eksplisit"] >= threshold]
        if query.strip():
            k = query.strip()
            view = view[view["Book-Title"].astype(str).str.contains(k, case=False, regex=False, na=False) |
                        view["Book-Author"].astype(str).str.contains(k, case=False, regex=False, na=False) |
                        view["Publisher"].astype(str).str.contains(k, case=False, regex=False, na=False)]
        view = view.sort_values([order, "Rating eksplisit", "ISBN"], ascending=[False, False, True])
        note(f"<b>{num(len(view))}</b> ISBN memenuhi filter. <b>Skor berbobot</b> = (n × rata-rata buku + {E['weight_m']} × rata-rata katalog "
             f"{num(E['catalog_mean'], 2)}) ÷ (n + {E['weight_m']}). Buku dengan sedikit rating ditarik ke rata-rata, sehingga skor "
             "tinggi dari 2-3 rating tidak mendominasi.")
        if view.empty:
            st.info("Tidak ada buku yang memenuhi filter. Turunkan batas rating atau ubah kata pencarian.")
        else:
            top = view.dropna(subset=[order]).head(top_n)
            lbl = (top["Book-Title"].astype(str).str.slice(0, 38) + " · " + top["ISBN"].str[-4:]).tolist()
            vals = top[order].round(2).tolist()
            fmt = (lambda v: f"{v:,.0f}") if order in ("Interaksi", "Rating eksplisit") else (lambda v: f"{v:.2f}")
            fig = hbar(lbl, vals, h=max(320, 34 * len(top)), fmt=fmt)
            if order in ("Skor berbobot", "Rata-rata"):
                fig.update_xaxes(range=[0, 11.5])
            a, b = st.columns([1.25, 1])
            with a:
                chart_card(f"Peringkat berdasarkan {order.lower()}", fig, "satu baris = satu ISBN (edisi) · 4 digit terakhir ISBN di label")
            with b:
                shown = view.head(top_n)
                with st.container(border=True):
                    st.markdown('<div class="ct">Sampul teratas</div><div class="cs">klik judul di Katalog untuk detail</div>', unsafe_allow_html=True)
                    shelf(shown, min(top_n, 8))
            cols = ["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication", "Interaksi", "Rating eksplisit", "Rata-rata", "Skor berbobot"]
            st.dataframe(view[cols].head(500), hide_index=True, use_container_width=True, height=360,
                         column_config={"Book-Title": st.column_config.TextColumn("Judul", width="large"),
                                        "Book-Author": "Penulis", "Year-Of-Publication": st.column_config.NumberColumn("Tahun", format="%d"),
                                        "Rata-rata": st.column_config.ProgressColumn("Rata-rata", min_value=0, max_value=10, format="%.2f"),
                                        "Skor berbobot": st.column_config.ProgressColumn("Skor berbobot", min_value=0, max_value=10, format="%.2f")})
            st.download_button("⬇️  Unduh hasil filter (CSV)", view[cols].to_csv(index=False).encode("utf-8-sig"),
                               "eksplorasi_buku.csv", "text/csv", key="eda_download")
            st.caption("Tabel menampilkan maksimal 500 baris; unduhan berisi seluruh hasil filter.")
        a, b = st.columns(2)
        with a:
            au = E["authors"]
            chart_card("Penulis dengan pembaca terbanyak", hbar([r["author"] for r in au], [r["readers"] for r in au], fmt=short),
                       "jumlah pembaca unik yang memberi rating eksplisit")
        with b:
            pub = E["publishers"]
            chart_card("Penerbit dengan pembaca terbanyak", hbar([r["publisher"] for r in pub], [r["readers"] for r in pub], fmt=short),
                       "penerbit ikut menjadi fitur teks TF-IDF")
        dc = E["decades"]
        if dc:
            top_d = max(dc, key=dc.get)
            chart_card("Tahun terbit per dekade", bar([f"{k}-an" for k in dc], list(dc.values()), highlight=f"{top_d}-an", h=280, fmt=short),
                       f"rentang 1950-2004 · {num(E['year_invalid'])} ISBN dengan tahun kosong/di luar rentang tidak diplot",
                       f"Katalog didominasi terbitan <b>{top_d}-an</b>. Tahun terbit bukan waktu membaca: data dikumpulkan tahun 2004.")

    # ---------- Profil pembaca ----------
    with t3:
        a, b = st.columns(2)
        with a:
            ah = E["age_hist"]
            if sum(ah.values()):
                x = [int(k) for k in ah]
                fig = go.Figure(go.Bar(x=[k + 2.5 for k in x], y=list(ah.values()), width=4.4,
                                       marker=dict(color=[GOLD if 25 <= k < 40 else SAGE for k in x], cornerradius=5),
                                       hovertemplate="usia %{x}<br><b>%{y:,}</b><extra></extra>"))
                fig.add_vline(x=E["age_median"], line_dash="dot", line_color=BURGUNDY,
                              annotation_text=f"median {E['age_median']:.0f}", annotation_font_color=BURGUNDY)
                chart_card("Distribusi usia", style(fig), f"{num(E['age_valid'])} profil dengan usia 5-100 tahun",
                           f"Usia kosong pada <b>{E['age_missing'] * 100:.0f}%</b> profil dan {num(E['age_out'])} usia di luar batas. "
                           "Profil tetap dihitung; hanya usianya yang tidak diplot. K-Prototypes harus menangani usia kosong.")
            else:
                st.info("Usia tidak tersedia pada cakupan ini.")
        with b:
            cc = E["countries"]
            chart_card("Negara asal pembaca", hbar(list(cc), list(cc.values()), fmt=short),
                       "dari kolom Country atau bagian terakhir Location · alias seperti 'us'/'usa' disatukan",
                       f"USA mendominasi ({list(cc.values())[0] / max(E['n_users'], 1) * 100:.0f}% profil). {num(E['country_unknown'])} profil "
                       "tanpa negara jelas. Negara kecil digabung menjadi 'Other' saat clustering.")

    # ---------- Kualitas & cakupan ----------
    with t4:
        a, b = st.columns([1, 1])
        with a:
            with st.container(border=True):
                st.markdown('<div class="ct">Cakupan data</div><div class="cs">baris 1 terhadap interaksi valid · baris 2-3 terhadap ISBN katalog</div>',
                            unsafe_allow_html=True)
                out = ""
                for (lbl, p), col in zip(E["coverage"].items(), [PINE, GOLD, BURGUNDY]):
                    out += (f'<div style="margin:.75rem 0"><div style="display:flex;justify-content:space-between;font-size:.85rem">'
                            f'<span>{lbl}</span><b style="color:{FOREST}">{p:.1f}%</b></div><div style="height:10px;background:#efe8d8;'
                            f'border-radius:6px;margin-top:.3rem"><div style="width:{max(p, 1.5):.1f}%;height:100%;background:{col};border-radius:6px"></div></div></div>')
                st.markdown(out + f'<div class="ins">💡 Kepadatan matriks eksplisit: <b>{E["density"]:.4f}%</b> pasangan pembaca–buku. '
                                  'Interaksi tanpa metadata buku tidak bisa direkomendasikan karena tidak punya teks untuk TF-IDF.</div>',
                            unsafe_allow_html=True)
        with b:
            with st.container(border=True):
                st.markdown('<div class="ct">Audit data</div><div class="cs">hal yang ditangani saat preprocessing</div>', unsafe_allow_html=True)
                out = ""
                for lbl, v in E["audit"].items():
                    ok = v == 0
                    out += (f'<div class="row" style="justify-content:space-between"><span style="font-size:.85rem">'
                            f'{"✅" if ok else "⚠️"} {lbl}</span><span class="tag" style="{"background:#e3efe4;color:#1f6b4f" if ok else ""}">'
                            f'{num(v)}</span></div>')
                st.markdown(out, unsafe_allow_html=True)
        heading("Implikasi untuk sistem rekomendasi", "Kesimpulan")
        imp = [("#1f6b4f", "Pisahkan sinyal", "Rating 0 adalah interaksi implisit, bukan skor buruk. SVD hanya dilatih pada rating 1-10."),
               ("#e3a83b", "Tampilkan bukti", "Bintang selalu disertai jumlah rating. Skor tinggi dari sedikit rating perlu dibaca hati-hati."),
               ("#8c2f39", "Siapkan cold-start", "Banyak profil tanpa riwayat: pengguna baru dilayani form preferensi (CBF) dan segmen K-Prototypes."),
               ("#0f3d2e", "Seimbangkan filter", "Filter pembaca dan buku memadatkan matriks, tetapi mengurangi cakupan. CBF menutup celah itu.")]
        for c, (col, t, d) in zip(st.columns(4), imp):
            c.markdown(f'<div class="step" style="--c:{col}"><b>{t}</b><p>{d}</p></div>', unsafe_allow_html=True)
        st.caption("Data tidak berisi waktu interaksi, pembelian, atau pendapatan, sehingga halaman ini tidak menyimpulkan tren waktu.")

# =========================================================
# EVALUASI & SEGMENTASI
# =========================================================
elif page == "Evaluasi & Segmentasi":
    page_title("Evaluasi & segmentasi", "Dua analisis terpisah: kualitas mesin hybrid, dan segmen pembaca untuk pengguna baru.")
    t1, t2 = st.tabs(["🎯  Evaluasi model hybrid", "👥  Segmentasi K-Prototypes"])

    with t1:
        cmp_df, al = D["cmp"], D["alpha"].sort_values("alpha")
        best = cmp_df.sort_values("RMSE").iloc[0]
        bp = al.loc[al["NDCG@10"].idxmax()] if "NDCG@10" in al else al.iloc[0]
        kpis([("🏆", "g1", "Model CF terpilih", str(best["Model"]), f"RMSE {num(best['RMSE'], 3)} · MAE {num(best['MAE'], 3)}"),
              ("⚖️", "g2", "α terbaik", num(bp["alpha"], 1), f"CF {bp['alpha'] * 100:.0f}% · CBF {(1 - bp['alpha']) * 100:.0f}%"),
              ("📐", "g3", "NDCG@10", num(bp.get("NDCG@10", 0), 3), "kualitas urutan top-10"),
              ("🎯", "g4", "Hit Rate@10", f"{bp.get('HitRate@10', 0) * 100:.1f}%", f"{int(bp.get('Evaluated Users', 0))} pembaca uji")])
        a, b = st.columns([1, 1.5])
        with a:
            fig = go.Figure()
            for met, col in [("RMSE", FOREST), ("MAE", GOLD)]:
                fig.add_bar(x=cmp_df["Model"], y=cmp_df[met], name=met, marker=dict(color=col, cornerradius=8),
                            text=[f"{v:.3f}" for v in cmp_df[met]], textposition="outside", cliponaxis=False)
            fig = style(fig, 330, legend=True)
            chart_card("Perbandingan model CF", fig, "error prediksi rating 1-10 · semakin rendah semakin baik",
                       f"<b>{best['Model']}</b> lebih akurat. Rata-rata prediksinya meleset sekitar {best['MAE']:.2f} poin dari rating asli.")
        with b:
            opts = [c for c in ["NDCG@10", "HitRate@10", "Recall@10", "Precision@10"] if c in al.columns]
            m = st.radio("Metrik", opts, horizontal=True, label_visibility="collapsed", key="w_radio_metrik")
            bm = al.loc[al[m].idxmax()]
            fig = go.Figure(go.Scatter(x=al["alpha"], y=al[m], mode="lines+markers", line=dict(color=PINE, width=3, shape="spline"),
                                       fill="tozeroy", fillcolor="rgba(31,107,79,.08)", marker=dict(size=8, color=PINE),
                                       hovertemplate="α %{x}<br><b>%{y:.4f}</b><extra></extra>"))
            fig.add_scatter(x=[bm["alpha"]], y=[bm[m]], mode="markers+text", text=[f"terbaik α={bm['alpha']:.1f}"],
                            textposition="top center", textfont=dict(color=BURGUNDY, size=11),
                            marker=dict(size=16, color=GOLD, line=dict(width=2, color=FOREST)))
            fig = style(fig, 300)
            fig.update_xaxes(title="α (bobot CF)", dtick=.1)
            chart_card(f"{m} pada tiap nilai α", fig, "hybrid = α·CF + (1−α)·CBF",
                       "Kinerja stabil di α 0,1-0,6 lalu turun tajam saat CF mendominasi (α ≥ 0,7). Pada data yang jarang ini, "
                       "<b>kemiripan konten</b> adalah sinyal terkuat, sedangkan CF memberi penyesuaian personal.")
        with st.expander("📘  Cara membaca metrik & rumus"):
            st.latex(r"S_{hybrid}(u,i)=\alpha\,S_{CF}(u,i)+(1-\alpha)\,S_{CBF}(u,i),\quad S_{CF}=\frac{\hat r_{ui}-1}{9}")
            st.markdown("- **RMSE / MAE**: rata-rata selisih prediksi SVD dengan rating asli (skala 1-10).\n"
                        "- **Precision@10**: porsi dari 10 rekomendasi yang memang disukai pembaca di data uji.\n"
                        "- **Recall@10 / Hit Rate@10**: seberapa sering buku yang disukai muncul di top-10.\n"
                        "- **NDCG@10**: seperti Hit Rate, tetapi memberi nilai lebih jika buku yang tepat berada di urutan atas.")
            st.dataframe(cmp_df, hide_index=True, use_container_width=True)
            st.dataframe(al, hide_index=True, use_container_width=True)

    with t2:
        cu, pb = D["cu"], D["pb"]
        if cu is None or pb is None:
            st.warning("Tambahkan cluster_users.csv dan popular_books.csv.")
        else:
            note("Segmentasi memakai <b>K-Prototypes</b> (gabungan K-Means untuk usia dan K-Modes untuk negara). "
                 "Hasilnya hanya dipakai sebagai rekomendasi awal untuk pengguna baru, bukan bagian dari skor hybrid.")
            cu = cu.assign(Age=pd.to_numeric(cu["Age"], errors="coerce"), Cluster=pd.to_numeric(cu["Cluster"], errors="coerce"))
            prof = cu.groupby("Cluster").agg(n=("User-ID", "size"), age=("Age", "mean"), med=("Age", "median"),
                                              ctry=("Country_Cluster", lambda s: s.mode().iat[0]))
            cl_cols = [FOREST, GOLD, BURGUNDY, SAGE, "#c9b48a", PINE]
            items = []
            for k, (cid, r) in enumerate(prof.iterrows()):
                items.append(("👥" if k % 2 == 0 else "🧓", ["g1", "g2", "g3", "g4"][k % 4], f"Cluster {int(cid)}",
                              short(r.n), f"usia rata-rata {r.age:.0f} th · {r.ctry}"))
            kpis(items)
            a, b = st.columns([1, 1.5])
            with a:
                chart_card("Ukuran segmen", donut([f"Cluster {int(c)}" for c in prof.index], prof["n"].tolist(),
                                                  cl_cols[:len(prof)], f"{short(len(cu))}<br><span style='font-size:11px'>pengguna</span>"))
            with b:
                fig = go.Figure()
                for k, cid in enumerate(prof.index):
                    ag = cu.loc[(cu["Cluster"] == cid) & cu["Age"].between(5, 100), "Age"]
                    fig.add_histogram(x=ag, name=f"Cluster {int(cid)}", xbins=dict(start=5, end=100, size=5),
                                      marker=dict(color=cl_cols[k % len(cl_cols)], line=dict(width=0)), opacity=.8)
                fig.update_layout(barmode="overlay")
                chart_card("Usia adalah pembeda utama segmen", style(fig, 300, legend=True), "histogram usia per cluster",
                           "Segmen terbagi tegas berdasarkan usia. Negara kurang membedakan karena USA mendominasi kedua segmen.")
            top_c = (cu[~cu["Country_Cluster"].isin(["Unknown"])].groupby(["Cluster", "Country_Cluster"]).size()
                     .groupby(level=0, group_keys=False).apply(lambda s: s / s.sum() * 100).rename("pct").reset_index())
            keep = top_c.groupby("Country_Cluster")["pct"].mean().nlargest(7).index
            fig = go.Figure()
            for k, cid in enumerate(prof.index):
                d = top_c[(top_c["Cluster"] == cid) & top_c["Country_Cluster"].isin(keep)].set_index("Country_Cluster").reindex(keep)
                fig.add_bar(x=list(keep), y=d["pct"], name=f"Cluster {int(cid)}", marker=dict(color=cl_cols[k % len(cl_cols)], cornerradius=6),
                            hovertemplate="%{x}<br><b>%{y:.1f}%</b><extra></extra>")
            fig.update_layout(barmode="group")
            fig.update_yaxes(ticksuffix="%")
            chart_card("Komposisi negara per segmen", style(fig, 300, legend=True), "% anggota cluster")
            heading("Buku favorit tiap segmen", "Top 5", "berdasarkan Bayesian Weighted Rating dalam cluster")
            for cid in prof.index:
                st.markdown(f"**Cluster {int(cid)}** · usia rata-rata {prof.loc[cid, 'age']:.0f} tahun")
                grid(pb[pd.to_numeric(pb["Cluster"], errors="coerce") == cid].sort_values("Skor_Buku", ascending=False).head(5),
                     f"seg{cid}", score="Skor_Buku", score_label="Skor segmen", rank=True)
