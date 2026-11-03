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

    D["eda"] = json.load(open(BASE / "eda_summary.json", encoding="utf-8")) if (BASE / "eda_summary.json").exists() else None
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
    top = cand.sort_values("Skor Hybrid", ascending=False).head(top_n).reset_index(drop=True)
    top["Karena"] = because(top["ISBN"].tolist(), liked["ISBN"].tolist())
    return top


@st.cache_data(show_spinner="Mencari buku yang mirip...")
def recommend_from_books(isbns, top_n):
    prof = profile(list(isbns))
    if prof is None:
        return pd.DataFrame()
    s = cosine_similarity(prof, TFIDF).ravel()
    cand = BOOKS.assign(**{"Skor CBF": s})
    cand = cand[~cand["ISBN"].isin(isbns)].sort_values("Skor CBF", ascending=False).head(top_n).reset_index(drop=True)
    cand["Karena"] = because(cand["ISBN"].tolist(), list(isbns))
    return cand


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
            if why:
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
                f'<br><br><b>Pengguna baru</b><br>CBF dari buku pilihan, atau K-Prototypes dari usia &amp; negara.'
                f'<br><br><b>★ Bintang</b><br>Rating umum 1-10 seluruh pembaca, dikonversi ke 5 bintang.</div>',
                unsafe_allow_html=True)

page = ss.page
E = D["eda"]

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

    heading("Bagaimana rekomendasi dibuat", "Cara kerja", "satu mesin utama, dua jalur untuk pengguna baru")
    s = st.columns(4)
    steps = [("#1f6b4f", "01 · CF", "Collaborative Filtering", "SVD mempelajari pola rating ribuan pembaca untuk memprediksi rating buku yang belum dibaca."),
             ("#e3a83b", "02 · CBF", "Content-Based", "TF-IDF judul, penulis, penerbit, tahun. Buku yang mirip dengan favoritmu mendapat skor tinggi."),
             ("#0f3d2e", "03 · HYBRID", "Skor gabungan", f"Skor = α·CF + (1−α)·CBF, dengan α = {BEST_ALPHA:.1f} hasil evaluasi NDCG@10."),
             ("#8c2f39", "04 · BARU", "Pengguna baru", "Pilih buku favorit (CBF), atau isi usia dan negara untuk rekomendasi dari segmen K-Prototypes.")]
    for c, (col, n, t, d) in zip(s, steps):
        c.markdown(f'<div class="step" style="--c:{col}"><div class="n">{n}</div><b>{t}</b><p>{d}</p></div>', unsafe_allow_html=True)

    heading("Rekomendasi untukmu", "Personal", "skor hybrid untuk pembaca terpilih")
    c1, c2 = st.columns([1.1, 3])
    with c1:
        u = user_picker("home_user")
        hist = RATINGS[RATINGS["User-ID"] == u]
        st.markdown(f'<div class="note g">Pembaca <b>{u}</b> sudah menilai <b>{len(hist)}</b> buku dengan rata-rata '
                    f'<b>{num(hist["Book-Rating"].mean(), 1)}/10</b>.</div>', unsafe_allow_html=True)
        st.button("Lihat semua rekomendasi →", on_click=go_page, args=("Rekomendasi",), use_container_width=True)
    with c2:
        recs = recommend(u, BEST_ALPHA, 4, LIKE_THR)
        if recs.empty:
            st.info("Belum ada rekomendasi untuk pembaca ini.")
        else:
            grid(recs, "home", ncols=4, score="Skor Hybrid", rank=True)

    heading("Buku populer", "Trending", "dari seluruh rating eksplisit Book-Crossing")
    a, b = st.columns([3, 1])
    mode = a.radio("Urutkan", ["🔥 Paling banyak dinilai", "⭐ Rating tertinggi"], horizontal=True, label_visibility="collapsed")
    n_show = b.select_slider("Jumlah", [5, 10, 15], value=5, label_visibility="collapsed")
    pop = CAT[CAT["in_model"]]
    if mode.startswith("🔥"):
        pop = pop.sort_values("n", ascending=False)
    else:
        pop = pop[pop["n"] >= 30].sort_values("bayes", ascending=False)
        st.caption("Diurutkan dengan Bayesian Weighted Rating (minimal 30 rating) agar buku dengan sedikit rating tidak mendominasi.")
    grid(pop.head(n_show), "pop", rank=True)

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
    page_title("Rekomendasi buku", "Pilih jalur sesuai kondisi pembaca: sudah punya riwayat atau pengguna baru.")
    t1, t2, t3 = st.tabs(["👤  Punya User-ID · Hybrid", "📖  Pengguna baru · Pilih buku favorit", "🌍  Pengguna baru · Usia & negara"])

    # ---- Hybrid ----
    with t1:
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        with c1:
            u = user_picker("rec_user")
        top_n = c2.select_slider("Jumlah buku", [5, 10, 15, 20], value=10)
        alpha = c3.slider("α (bobot CF)", 0.0, 1.0, float(np.clip(BEST_ALPHA, 0, 1)), 0.05,
                          help=f"0 = murni konten, 1 = murni CF. Nilai terbaik hasil evaluasi: {BEST_ALPHA}.")
        thr = c4.slider("Ambang 'disukai'", 1, 10, LIKE_THR, help="Rating minimal agar buku dianggap disukai (membentuk profil CBF).")

        hist = RATINGS[RATINGS["User-ID"] == u].sort_values("Book-Rating", ascending=False)
        liked = liked_of(hist, thr)
        top_au = enrich(hist)["Book-Author"].value_counts()
        kpis([("📚", "g1", "Buku dinilai", num(len(hist)), "riwayat di data latih"),
              ("⭐", "g2", "Rata-rata rating", f"{num(hist['Book-Rating'].mean(), 1)}/10", f"≈ {num(hist['Book-Rating'].mean() / 2, 1)} bintang"),
              ("❤️", "g3", "Buku disukai", num(len(liked)), f"rating ≥ {thr} → profil konten"),
              ("✍️", "g4", "Penulis favorit", html.escape(str(top_au.index[0])[:18]) if len(top_au) else "-",
               f"{top_au.iloc[0]} buku dinilai" if len(top_au) else "")])

        recs = recommend(u, alpha, top_n, thr)
        left, right = st.columns([3.3, 1])
        with right:
            heading("Rak favoritnya", "Profil", "buku dengan rating tertinggi")
            with st.container(border=True):
                shelf(hist, 6)
        with left:
            heading(f"Top {top_n} untuk pembaca {u}", "Hybrid CF + CBF",
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
                               .to_csv(index=False).encode("utf-8-sig"), f"rekomendasi_user_{u}.csv", "text/csv")

    # ---- Pengguna baru: pilih buku ----
    with t2:
        note("Belum punya User-ID? Pilih <b>1-5 buku yang kamu sukai</b>. Sistem membentuk profil konten dari buku itu "
             "lalu mencari buku dengan judul, penulis, dan penerbit yang paling mirip (Content-Based Filtering). "
             "CF tidak dipakai karena model SVD belum mengenal pengguna baru.", green=True)
        opts = CAT[CAT["in_model"]].sort_values("n", ascending=False)
        label = dict(zip(opts["ISBN"], opts["Book-Title"].str.slice(0, 70) + " — " + opts["Book-Author"].astype(str)))
        a, b = st.columns([3, 1])
        picks = a.multiselect("Buku yang kamu sukai", list(label), format_func=label.get, max_selections=5,
                              placeholder="Ketik judul atau penulis...", default=list(label)[:0])
        n_new = b.select_slider("Jumlah rekomendasi", [5, 10, 15], value=10, key="nn")
        if not picks:
            st.caption("Belum ada pilihan. Contoh: pilih beberapa novel Harry Potter atau John Grisham.")
            heading("Atau mulai dari buku terpopuler", "Inspirasi")
            grid(opts.head(5), "insp")
        else:
            with st.container(border=True):
                st.markdown('<div class="ct">Buku pilihanmu</div>', unsafe_allow_html=True)
                cols = st.columns(len(picks))
                for c, i in zip(cols, picks):
                    r = D["bidx"].loc[i]
                    c.markdown(f'<div class="row">{cover(r.get("Image-URL-M"), r["Book-Title"], "")}<div><b>'
                               f'{html.escape(str(r["Book-Title"])[:40])}</b><small>{html.escape(str(r["Book-Author"]))}</small></div></div>',
                               unsafe_allow_html=True)
            nr = recommend_from_books(tuple(picks), n_new)
            heading("Rekomendasi untukmu", "Content-Based", "diurutkan dari kemiripan konten tertinggi")
            grid(nr, "nb", score="Skor CBF", score_label="Kemiripan", rank=True, why=True)

    # ---- Pengguna baru: cluster ----
    with t3:
        note("Tidak ingin memilih buku? Isi <b>usia dan negara</b>. Model K-Prototypes menempatkanmu ke segmen pembaca "
             "yang mirip, lalu menampilkan buku favorit segmen itu (diurutkan dengan Bayesian Weighted Rating). "
             "Jalur ini terpisah dan <b>tidak</b> ikut dalam skor hybrid.")
        need = ["kproto_final.joblib", "cluster_users.csv", "popular_books.csv"]
        miss = [f for f in need if not (BASE / f).exists()]
        if miss:
            st.warning("Artefak clustering belum ada: " + ", ".join(miss))
        else:
            try:
                kp = joblib.load(BASE / need[0])
                cu, pb = D["cu"], D["pb"]
                ctry = sorted(c for c in cu["Country_Cluster"].dropna().astype(str).unique() if c not in ("Other", "Unknown", "", "nan", "None"))
                c1, c2, c3 = st.columns(3)
                age = c1.slider("Usia", 5, 100, 27)
                country = c2.selectbox("Negara", ctry + ["Lainnya"], index=ctry.index("USA") if "USA" in ctry else 0)
                n_c = c3.select_slider("Jumlah buku", [5, 10, 15, 20], value=10)
                cval = country if country in ctry else "Other"
                cl = int(kp.predict(np.array([[float(age), cval]], dtype=object), categorical=[1])[0])
                prof = cu[pd.to_numeric(cu["Cluster"], errors="coerce") == cl]
                pa = pd.to_numeric(prof["Age"], errors="coerce")
                cb = pb[pd.to_numeric(pb["Cluster"], errors="coerce") == cl].dropna(subset=["Book-Title"])
                kpis([("🧭", "g1", "Segmen kamu", f"Cluster {cl}", "hasil K-Prototypes"),
                      ("👥", "g4", "Anggota segmen", short(len(prof)), f"{len(prof) / len(cu) * 100:.0f}% dari semua pengguna"),
                      ("🎂", "g2", "Usia rata-rata", f"{pa.mean():.0f} th", f"median {pa.median():.0f} th"),
                      ("🌍", "g3", "Negara dominan", str(prof["Country_Cluster"].mode().iat[0]) if len(prof) else "-",
                       f"{(prof['Country_Cluster'] == prof['Country_Cluster'].mode().iat[0]).mean() * 100:.0f}% anggota" if len(prof) else "")])
                if cb.empty:
                    st.warning("Belum ada buku untuk segmen ini.")
                else:
                    heading(f"Favorit segmen Cluster {cl}", "K-Prototypes",
                            "skor segmen = Bayesian Weighted Rating dalam cluster · bintang = rating umum")
                    grid(cb.sort_values(["Skor_Buku", "Jumlah_Rating"], ascending=False).head(n_c), "cl",
                         score="Skor_Buku", score_label="Skor segmen", rank=True)
            except Exception as exc:
                st.error(f"Gagal memuat model clustering: {exc}")
                st.caption("Periksa versi kmodes dan scikit-learn, serta urutan fitur (Age, Country_Cluster).")

# =========================================================
# KATALOG
# =========================================================
elif page == "Katalog":
    page_title("Katalog buku", "Cari, saring, dan jelajahi buku. Klik Detail untuk melihat buku yang serupa.")
    base = CAT[CAT["in_model"]].copy()
    with st.container(border=True):
        a, b, c, d = st.columns([2.2, 1.4, 1.1, 1.1])
        q = a.text_input("Cari", placeholder="🔎  Judul, penulis, atau penerbit...", label_visibility="collapsed")
        authors = ["Semua penulis"] + base["Book-Author"].value_counts().index.tolist()
        au = b.selectbox("Penulis", authors, label_visibility="collapsed")
        min_s = c.select_slider("Bintang minimal", [0, 3, 3.5, 4, 4.5], value=0, format_func=lambda v: "Semua ★" if v == 0 else f"≥ {v} ★",
                                label_visibility="collapsed")
        sort = d.selectbox("Urutkan", ["Terpopuler", "Bintang tertinggi", "Terbaru", "Judul A-Z"], label_visibility="collapsed")
        y = base["Year-Of-Publication"]
        yr = st.slider("Tahun terbit", 1950, 2005, (1950, 2005))
    m = (y.isna() | y.between(*yr) | (y <= 0)) if yr == (1950, 2005) else y.between(*yr)
    if q.strip():
        k = q.strip()
        m &= (base["Book-Title"].str.contains(k, case=False, regex=False, na=False) |
              base["Book-Author"].str.contains(k, case=False, regex=False, na=False) |
              base["Publisher"].astype(str).str.contains(k, case=False, regex=False, na=False))
    if au != "Semua penulis":
        m &= base["Book-Author"] == au
    if min_s:
        m &= base["avg"] / 2 >= min_s
    res = base[m]
    res = {"Terpopuler": res.sort_values("n", ascending=False),
           "Bintang tertinggi": res.sort_values("bayes", ascending=False),
           "Terbaru": res.sort_values("Year-Of-Publication", ascending=False),
           "Judul A-Z": res.sort_values("Book-Title")}[sort]
    per = 18
    pages = max(1, int(np.ceil(len(res) / per)))
    h1, h2 = st.columns([3, 1])
    h1.markdown(f'<div class="sec" style="margin-top:.8rem"><span><b style="display:inline;font-size:1.3rem">{num(len(res))}</b> buku ditemukan</span></div>',
                unsafe_allow_html=True)
    pg = h2.number_input("Halaman", 1, pages, 1, help=f"dari {pages} halaman") if pages > 1 else 1
    if res.empty:
        st.warning("Tidak ada buku yang cocok. Coba longgarkan filter.")
    else:
        grid(res.iloc[(pg - 1) * per: pg * per], f"cat{pg}", ncols=6)

# =========================================================
# EKSPLORASI DATA
# =========================================================
elif page == "Eksplorasi Data":
    page_title("Eksplorasi data", "Hanya analisis yang memengaruhi kualitas rekomendasi. Dihitung dari data mentah Book-Crossing.")
    if not E:
        st.warning("Tambahkan eda_summary.json (jalankan prep_eda.py) untuk menampilkan halaman ini.")
        st.stop()
    exp_share = E["n_explicit"] / E["n_ratings"]
    kpis([("🗂️", "g1", "Total interaksi", short(E["n_ratings"]), f"{num(E['n_users'])} pengguna · {short(E['n_books'])} buku"),
          ("⭐", "g2", "Rating eksplisit", short(E["n_explicit"]), f"{exp_share * 100:.0f}% dari interaksi"),
          ("📈", "g3", "Rata-rata rating", f"{num(E['mean_rating'], 2)}/10", f"≈ {num(E['mean_rating'] / 2, 1)} ★"),
          ("🕳️", "g4", "Matriks kosong", f"{E['sparsity'] * 100:.3f}%", "pasangan pembaca-buku tanpa rating")])

    t1, t2, t3, t4 = st.tabs(["⭐  Rating", "📉  Aktivitas & sparsity", "👥  Pembaca", "📚  Katalog & kualitas data"])
    with t1:
        a, b = st.columns([1, 1.5])
        with a:
            chart_card("Interaksi implisit vs eksplisit",
                       donut(["Implisit (rating 0)", "Eksplisit (1-10)"], [E["n_implicit"], E["n_explicit"]], [SAGE, FOREST],
                             f"{exp_share * 100:.0f}%<br><span style='font-size:11px'>eksplisit</span>"),
                       "rating 0 = interaksi tanpa nilai",
                       f"<b>{E['n_implicit'] / E['n_ratings'] * 100:.0f}%</b> interaksi tidak memberi nilai, jadi dibuang dari pelatihan SVD "
                       "yang memprediksi rating 1-10.")
        with b:
            rd = E["rating_dist"]
            ks = list(rd.keys())
            mx = max(rd, key=rd.get)
            chart_card("Distribusi rating eksplisit", bar(ks, list(rd.values()), highlight=mx, h=300, fmt=short),
                       "jumlah rating per nilai 1-10",
                       f"Rating <b>{mx}</b> paling sering. {sum(v for k, v in rd.items() if int(k) >= 7) / sum(rd.values()) * 100:.0f}% rating bernilai ≥ 7: "
                       "pembaca cenderung hanya menilai buku yang mereka suka. Ini disebut <b>positivity bias</b>.")
        s5 = {str(i): 0 for i in range(1, 6)}
        for k, v in rd.items():
            s5[str(int(np.ceil(int(k) / 2)))] += v
        with st.container(border=True):
            st.markdown('<div class="ct">Dikonversi ke 5 bintang (seperti di kartu buku)</div>', unsafe_allow_html=True)
            tot = sum(s5.values())
            rows = ""
            for k in ["5", "4", "3", "2", "1"]:
                p = s5[k] / tot * 100
                rows += (f'<div style="display:flex;align-items:center;gap:.7rem;margin:.25rem 0"><span style="width:62px;color:#e3a83b">{"★" * int(k)}</span>'
                         f'<div style="flex:1;height:12px;background:#efe8d8;border-radius:8px;overflow:hidden">'
                         f'<div style="width:{p:.1f}%;height:100%;background:linear-gradient(90deg,#1f6b4f,#e3a83b);border-radius:8px"></div></div>'
                         f'<span style="width:110px;text-align:right;font-size:.8rem;color:#6f7d74">{p:.1f}% · {short(s5[k])}</span></div>')
            st.markdown(rows + '<div class="ins">💡 Rating 9-10 = 5★, 7-8 = 4★, 5-6 = 3★, 3-4 = 2★, 1-2 = 1★.</div>', unsafe_allow_html=True)

    with t2:
        a, b = st.columns(2)
        with a:
            ub = E["user_bins"]
            chart_card("Jumlah rating per pembaca", bar(list(ub), list(ub.values()), highlight="1", fmt=short),
                       "pembaca dengan rating eksplisit",
                       f"Median hanya <b>{E['median_user']:.0f}</b> rating per pembaca. Hanya <b>{num(E['users_ge5'])}</b> pembaca "
                       "punya ≥ 5 rating. Inilah alasan model dilatih pada pembaca dengan ≥ 20 rating.")
        with b:
            bb = E["book_bins"]
            chart_card("Jumlah rating per buku", bar(list(bb), list(bb.values()), highlight="1", fmt=short, color="#c9b48a"),
                       "long tail katalog",
                       f"<b>{E['books_once'] * 100:.0f}%</b> buku hanya dinilai sekali. CF sulit memprediksi buku seperti ini, "
                       "sehingga CBF diperlukan sebagai pelengkap.")
        n_mod = META.get("n_model_ratings", len(RATINGS))
        fig = go.Figure(go.Funnel(
            y=["Semua interaksi", "Rating eksplisit", "Data model (user ≥ 20, buku ≥ 10)"],
            x=[E["n_ratings"], E["n_explicit"], n_mod], textinfo="value+percent initial",
            marker=dict(color=[SAGE, PINE, GOLD]), connector=dict(fillcolor="#efe8d8")))
        fig = style(fig, 300)
        fig.update_yaxes(tickfont=dict(color=INK, size=12))
        chart_card("Dari data mentah ke data latih model", fig, "penyaringan agar CF punya cukup sinyal",
                   f"Setelah penyaringan tersisa <b>{num(META.get('n_model_users', 0))}</b> pembaca dan "
                   f"<b>{num(META.get('n_model_books', len(BOOKS)))}</b> buku. Matriks jauh lebih padat sehingga pola rating bisa dipelajari.")

    with t3:
        a, b = st.columns(2)
        with a:
            ah = E["age_hist"]
            x = [int(k) for k in ah]
            fig = go.Figure(go.Bar(x=[k + 2.5 for k in x], y=list(ah.values()), width=4.4,
                                   marker=dict(color=[GOLD if 25 <= k < 40 else SAGE for k in x], cornerradius=5),
                                   hovertemplate="usia %{x}<br><b>%{y:,}</b><extra></extra>"))
            fig.add_vline(x=E["age_median"], line_dash="dot", line_color=BURGUNDY,
                          annotation_text=f"median {E['age_median']:.0f}", annotation_font_color=BURGUNDY)
            chart_card("Sebaran usia pembaca", style(fig), "rentang 5-100 tahun",
                       f"Pembaca terbanyak berusia 25-40. <b>{E['age_missing'] * 100:.0f}%</b> pengguna tidak mengisi usia, "
                       "sehingga K-Prototypes perlu menangani data kosong.")
        with b:
            cc = E["countries"]
            tot = sum(cc.values())
            chart_card("10 negara asal pembaca", hbar(list(cc), list(cc.values()), fmt=short), "",
                       f"USA mendominasi ({list(cc.values())[0] / E['n_users'] * 100:.0f}% pengguna). Negara kecil digabung "
                       "menjadi 'Other' saat clustering.")

    with t4:
        a, b = st.columns(2)
        with a:
            au = E["authors"]
            chart_card("Penulis dengan pembaca terbanyak", hbar([r["author"] for r in au], [r["readers"] for r in au], fmt=short),
                       "jumlah pembaca unik yang memberi rating eksplisit")
        with b:
            pub = E["publishers"]
            chart_card("Penerbit dengan pembaca terbanyak", hbar([r["publisher"] for r in pub], [r["readers"] for r in pub], fmt=short),
                       "penerbit ikut menjadi fitur teks TF-IDF")
        a, b = st.columns([1.4, 1])
        with a:
            dc = E["decades"]
            chart_card("Tahun terbit per dekade", bar([f"{k}-an" for k in dc], list(dc.values()), highlight="1990-an", fmt=short), "",
                       "Katalog didominasi buku terbitan 1990-an. Data Book-Crossing dikumpulkan tahun 2004.")
        with b:
            with st.container(border=True):
                st.markdown('<div class="ct">Kualitas data</div><div class="cs">hal yang ditangani saat preprocessing</div>', unsafe_allow_html=True)
                q = [("ISBN rating ada di katalog", E["isbn_match"] * 100, f"{E['isbn_match'] * 100:.1f}%", PINE),
                     ("Usia pengguna kosong", E["age_missing"] * 100, f"{E['age_missing'] * 100:.1f}%", BURGUNDY),
                     ("Buku dengan ≥ 10 rating", E["books_ge10"] / E["n_books_explicit"] * 100, f"{num(E['books_ge10'])} buku", GOLD),
                     ("Tahun terbit tidak valid", E["year_invalid"] / E["n_books"] * 100, f"{num(E['year_invalid'])} buku", SAGE)]
                out = ""
                for lbl, p, txt, col in q:
                    out += (f'<div style="margin:.55rem 0"><div style="display:flex;justify-content:space-between;font-size:.82rem">'
                            f'<span>{lbl}</span><b style="color:{FOREST}">{txt}</b></div><div style="height:8px;background:#efe8d8;border-radius:6px;margin-top:.25rem">'
                            f'<div style="width:{max(p, 1.5):.1f}%;height:100%;background:{col};border-radius:6px"></div></div></div>')
                st.markdown(out + '<div class="ins">💡 Rating dengan ISBN di luar katalog tidak bisa direkomendasikan karena tidak punya metadata untuk TF-IDF.</div>',
                            unsafe_allow_html=True)

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
            m = st.radio("Metrik", opts, horizontal=True, label_visibility="collapsed")
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
