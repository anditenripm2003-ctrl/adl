import base64
import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sparse
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommender Dashboard", page_icon="📖", layout="wide",
                   initial_sidebar_state="expanded")

BASE = Path(__file__).resolve().parent
FILES = {"model": "cf_model.joblib", "tfidf": "tfidf_matrix.npz", "ratings": "model_ratings.pkl",
         "books": "candidate_books.pkl", "cmp": "cf_comparison.csv", "alpha": "alpha_evaluation.csv",
         "meta": "metadata.json"}
GD, GREEN, GOLD, INK, LINE = "#1b4d3a", "#3f7a4f", "#f2b01e", "#1d3a2f", "#eadfc8"

# =========================================================
# Gaya: krem, hijau tua, emas (mengikuti template hero)
# =========================================================
def hero_css():
    p = BASE / "hero_books.jpg"
    img = ("url(data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() + ") right center/auto 100% no-repeat,"
           if p.exists() else "")
    return img


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;1,600;1,700;1,800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],[data-testid="stAppViewContainer"]{font-family:'Plus Jakarta Sans',sans-serif;}
.stApp{background:#fbf5ea;} [data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.4rem;max-width:1400px;}
[data-testid="stSidebar"]{background:#1b4d3a;}
[data-testid="stSidebar"] button{justify-content:flex-start;border:0;border-radius:40px;padding:.6rem 1.1rem;font-weight:700;box-shadow:none;}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],[data-testid="stSidebar"] button[kind="secondary"]{background:transparent;color:#d8e6d6;}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover{background:rgba(255,255,255,.1);color:#fff;}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],[data-testid="stSidebar"] button[kind="primary"]{background:#f2b01e;color:#1b4d3a;}
.logo{font-family:'Playfair Display',serif;font-size:1.55rem;font-weight:800;color:#fbf5ea;margin:.3rem 0 1.4rem;line-height:1.1;}
.logo small{display:block;font-family:'Plus Jakarta Sans';font-size:.72rem;font-weight:500;color:#f2b01e;letter-spacing:.04em;}
/* HERO */
.st-key-hero{background:__IMG__ linear-gradient(90deg,#fdf6ea 0%,#fdf6ea 52%,rgba(253,246,234,0) 80%) , #fdf6ea;
  background:__IMG__ #fdf6ea;border-radius:30px;padding:2.2rem 2.8rem 2.6rem;position:relative;overflow:hidden;
  box-shadow:0 10px 30px rgba(27,77,58,.10);border-bottom:10px solid #1b4d3a;}
.st-key-hero:before{content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(90deg,#fdf6ea 0%,#fdf6ea 50%,rgba(253,246,234,.0) 78%);}
.st-key-hero>div{max-width:56%;position:relative;z-index:2;}
.pill{display:inline-block;background:rgba(255,255,255,.7);border:1px solid #eadfc8;border-radius:40px;padding:.4rem 1.2rem;color:#1b4d3a;font-size:1rem;}
.hl{font-family:'Playfair Display',serif;font-size:clamp(2.6rem,4.6vw,4.2rem);line-height:1.02;margin:1rem 0 .3rem;color:#1b4d3a;}
.hl i{font-weight:700;display:block;} .hl .m{font-style:normal;font-weight:800;display:block;}
.hl .g{color:#f2b01e;font-weight:800;} .hl .sp{color:#f2b01e;font-size:.45em;vertical-align:top;}
.uline{height:6px;width:260px;border-radius:6px;background:linear-gradient(90deg,#f2b01e,rgba(242,176,30,.1));margin:.1rem 0 1rem;}
.sub{color:#1b4d3a;font-size:1.05rem;line-height:1.55;max-width:430px;margin:0;}
.st-key-hero .stButton{margin-top:.4rem;}
.st-key-hero button{background:#1b4d3a;color:#fff;border:0;border-radius:50px;padding:.7rem 1.1rem .7rem 2rem;font-size:1.1rem;
  font-weight:700;box-shadow:0 8px 18px rgba(27,77,58,.28);}
.st-key-hero button:after{content:"→";display:inline-flex;align-items:center;justify-content:center;width:2.5rem;height:2.5rem;
  border-radius:50%;background:#f6c744;color:#1b4d3a;margin-left:2rem;font-size:1.3rem;}
.st-key-hero button:hover{background:#133a2b;color:#fff;}
.feat{display:flex;gap:1.4rem;margin-top:1rem;color:#1b4d3a;font-weight:600;font-size:.88rem;line-height:1.2;}
.feat div{display:flex;gap:.6rem;align-items:center;} .feat i{font-style:normal;background:#f1ecdc;border-radius:12px;padding:.55rem .65rem;font-size:1.2rem;}
/* seksi & kartu */
.sec{display:flex;gap:.8rem;align-items:baseline;margin:1.6rem 0 .5rem;}
.sec b{font-family:'Playfair Display',serif;font-size:1.65rem;font-weight:800;color:#1b4d3a;} .sec span{color:#7b8a80;font-size:.9rem;}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fffdf8;border-radius:20px;border-color:#eadfc8;}
.bk{text-align:center;} .bt{font-weight:800;font-size:.9rem;line-height:1.2;margin-top:.5rem;min-height:2.3em;color:#1b4d3a;}
.ba{color:#7b8a80;font-size:.78rem;} .bs{color:#b27f00;font-weight:800;font-size:.82rem;}
.cover{width:100%;max-width:112px;height:150px;object-fit:cover;border-radius:10px;box-shadow:0 8px 18px rgba(27,77,58,.22);display:block;margin:0 auto;}
.cover.ph{display:flex;align-items:center;justify-content:center;background:linear-gradient(160deg,#1b4d3a,#3f7a4f);color:#fbf5ea;
  font-weight:700;font-size:.78rem;padding:.5rem;box-sizing:border-box;}
.rk{display:inline-block;min-width:2.1rem;padding:.1rem .5rem;border-radius:20px;font-weight:800;font-size:.85rem;border:2px solid #3f7a4f;color:#1b4d3a;}
.rk.r1{background:#f2b01e;border-color:#f2b01e;} .rk.r2{background:#d9dedb;border-color:#c3cbc6;} .rk.r3{background:#e8b98a;border-color:#e0a770;}
.av{width:70px;height:70px;border-radius:50%;border:3px solid #f2b01e;margin:.5rem auto .3rem;display:flex;align-items:center;
  justify-content:center;font-family:'Playfair Display',serif;font-weight:800;font-size:1.4rem;color:#1b4d3a;background:#f1ecdc;}
.row{display:flex;gap:.7rem;align-items:center;padding:.4rem 0;} .row .cover{width:42px;height:60px;max-width:42px;margin:0;box-shadow:none;}
.row b{font-size:.9rem;display:block;color:#1b4d3a;} .row small{color:#7b8a80;}
.ct{font-weight:800;color:#1b4d3a;font-size:1.02rem;margin-bottom:.1rem;}
[data-testid="stMetric"]{background:#fffdf8;border:1px solid #eadfc8;border-radius:18px;padding:.9rem 1.1rem;}
[data-testid="stMetricLabel"]{color:#7b8a80;} [data-testid="stMetricValue"]{color:#1b4d3a;font-weight:800;}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid #d9cfb4;background:#fffdf8;color:#1b4d3a;font-weight:700;padding:.25rem .6rem;min-height:0;}
.stButton>button:hover{border-color:#1b4d3a;color:#1b4d3a;background:#f6efdc;}
[data-testid="stTextInput"] input{border-radius:30px;background:#fffdf8;border:1px solid #eadfc8;padding:.7rem 1.2rem;}
button[data-baseweb="tab"]{font-weight:700;} [data-baseweb="tab-highlight"]{background:#f2b01e;}
</style>
"""
st.markdown(CSS.replace("__IMG__", hero_css()), unsafe_allow_html=True)


# =========================================================
# Muat data
# =========================================================
@st.cache_resource(show_spinner="Memuat model dan data...")
def load():
    miss = [f for f in FILES.values() if not (BASE / f).exists()]
    if miss:
        raise FileNotFoundError("File belum ada: " + ", ".join(miss))
    D = {"cf": joblib.load(BASE / FILES["model"]), "tfidf": sparse.load_npz(BASE / FILES["tfidf"]),
         "ratings": pd.read_pickle(BASE / FILES["ratings"]), "books": pd.read_pickle(BASE / FILES["books"]),
         "cmp": pd.read_csv(BASE / FILES["cmp"]), "alpha": pd.read_csv(BASE / FILES["alpha"]),
         "meta": json.load(open(BASE / FILES["meta"], encoding="utf-8"))}
    r, b = D["ratings"], D["books"]
    r["User-ID"] = pd.to_numeric(r["User-ID"], errors="coerce")
    r["Book-Rating"] = pd.to_numeric(r["Book-Rating"], errors="coerce")
    r["ISBN"], b["ISBN"] = r["ISBN"].astype(str), b["ISBN"].astype(str)
    D["row"] = {x: i for i, x in enumerate(b["ISBN"])}
    D["bidx"] = b.drop_duplicates("ISBN").set_index("ISBN")
    g = r.groupby("ISBN").agg(n=("Book-Rating", "size"), avg=("Book-Rating", "mean"))
    D["stats"] = g
    D["pop"] = g.reset_index().merge(b.drop_duplicates("ISBN"), on="ISBN")
    rb = r.merge(D["bidx"][["Book-Author"]].reset_index(), on="ISBN")
    D["au"] = (rb.groupby("Book-Author").agg(Pembaca=("User-ID", "nunique"), Rating=("Book-Rating", "size"),
                                              Rata=("Book-Rating", "mean"))
               .sort_values(["Pembaca", "Rating"], ascending=False).head(20))
    cu = BASE / "cluster_users.csv"
    D["readers"] = None
    if cu.exists():
        u = pd.DataFrame({"User-ID": r["User-ID"].dropna().astype(int).unique()})
        c = pd.read_csv(cu)[["User-ID", "Age", "Country"]].drop_duplicates("User-ID")
        D["readers"] = u.merge(c, on="User-ID", how="left")
    return D


try:
    D = load()
except Exception as exc:
    st.error(f"Dashboard belum bisa dimuat: {exc}")
    st.stop()

CF, TFIDF, RATINGS, BOOKS, META = D["cf"], D["tfidf"], D["ratings"], D["books"], D["meta"]
ss = st.session_state
ss.setdefault("page", "Home")
ss.setdefault("fav", [])
ss.setdefault("author", None)
uids = sorted(RATINGS["User-ID"].dropna().astype(int).unique().tolist())
ss.setdefault("user", int(RATINGS.groupby("User-ID").size().idxmax()))


# =========================================================
# Logika rekomendasi
# =========================================================
def user_profile(hist, thr):
    hist = hist[hist["ISBN"].isin(D["row"])]
    if hist.empty:
        return None
    liked = hist[hist["Book-Rating"] >= thr]
    if liked.empty:
        liked = hist.nlargest(min(5, len(hist)), "Book-Rating")
    w = liked["Book-Rating"].to_numpy(float) / 10.0
    rows = [D["row"][i] for i in liked["ISBN"]]
    return sparse.csr_matrix(TFIDF[rows].multiply(w[:, None]).sum(axis=0) / max(w.sum(), 1e-9))


@st.cache_data(show_spinner="Menghitung rekomendasi...")
def recommend(uid, alpha, top_n, thr):
    hist = RATINGS[RATINGS["User-ID"] == int(uid)]
    cand = BOOKS[~BOOKS["ISBN"].isin(set(hist["ISBN"]))].copy()
    if hist.empty or cand.empty:
        return pd.DataFrame()
    isb = cand["ISBN"].tolist()
    est = np.array([CF.predict(int(uid), i).est for i in isb], float)
    cf_s = np.clip((est - 1) / 9, 0, 1)
    prof = user_profile(hist, thr)
    cbf = np.zeros(len(isb)) if prof is None else cosine_similarity(prof, TFIDF[[D["row"][i] for i in isb]]).ravel()
    cand["Prediksi CF"], cand["Skor CF"], cand["Skor CBF"] = est, cf_s, cbf
    cand["Skor Hybrid"] = alpha * cf_s + (1 - alpha) * cbf
    return cand.sort_values("Skor Hybrid", ascending=False).head(top_n).reset_index(drop=True)


def similar(isbn, n=5):
    i = D["row"].get(isbn)
    if i is None:
        return pd.DataFrame()
    s = cosine_similarity(TFIDF[i], TFIDF).ravel()
    s[i] = -1
    idx = np.argsort(-s)[:n]
    return BOOKS.iloc[idx]


# =========================================================
# Komponen UI
# =========================================================
def cover(url, title=""):
    if isinstance(url, str) and url.startswith("http"):
        return f'<img class="cover" src="{html.escape(url)}" onerror="this.style.display=\'none\'">'
    return f'<div class="cover ph">{html.escape(str(title)[:40])}</div>'


def toggle_fav(i):
    ss.fav.remove(i) if i in ss.fav else ss.fav.append(i)


def go_page(p):
    ss.page = p


def heading(t, extra=""):
    st.markdown(f'<div class="sec"><b>{t}</b><span>{extra}</span></div>', unsafe_allow_html=True)


def mini(r):
    return (f'<div class="row">{cover(r.get("Image-URL-S") or r.get("Image-URL-M"), "")}<div>'
            f'<b>{html.escape(str(r["Book-Title"])[:46])}</b><small>{html.escape(str(r["Book-Author"])[:30])}</small></div></div>')


@st.dialog("Detail buku", width="large")
def detail(isbn):
    if isbn not in D["bidx"].index:
        st.write("Data buku tidak ditemukan.")
        return
    info = D["bidx"].loc[isbn]
    a, b = st.columns([1, 2])
    a.markdown(cover(info.get("Image-URL-L") or info.get("Image-URL-M"), info.get("Book-Title")), unsafe_allow_html=True)
    s = D["stats"].loc[isbn] if isbn in D["stats"].index else None
    b.markdown(f"### {info.get('Book-Title', '')}")
    b.write(f"**{info.get('Book-Author', '')}**  \n{info.get('Publisher', '')} · {info.get('Year-Of-Publication', '')}")
    if s is not None:
        b.write(f"⭐ {s['avg']:.1f} dari {int(s['n'])} rating")
    b.button("♥ Hapus dari favorit" if isbn in ss.fav else "♡ Simpan ke favorit", key=f"dfav{isbn}",
             on_click=toggle_fav, args=(isbn,))
    st.markdown("**Buku mirip**")
    for _, r in similar(isbn).iterrows():
        st.markdown(mini(r), unsafe_allow_html=True)


def grid(df, key, ncols=5, score=None):
    df = df.reset_index(drop=True)
    for s in range(0, len(df), ncols):
        for c, (i, row) in zip(st.columns(ncols), df.iloc[s:s + ncols].iterrows()):
            isbn, t = str(row["ISBN"]), str(row.get("Book-Title", ""))
            sub = f'<div class="bs">⭐ {row[score]:.3f}</div>' if score and pd.notna(row.get(score)) else ""
            with c, st.container(border=True):
                st.markdown(f'<div class="bk">{cover(row.get("Image-URL-M"), t)}<div class="bt">{html.escape(t[:46])}</div>'
                            f'<div class="ba">{html.escape(str(row.get("Book-Author", ""))[:28])}</div>{sub}</div>',
                            unsafe_allow_html=True)
                st.button("♥ Tersimpan" if isbn in ss.fav else "♡ Simpan", key=f"{key}f{i}{isbn}",
                          on_click=toggle_fav, args=(isbn,), use_container_width=True)
                if st.button("Detail", key=f"{key}d{i}{isbn}", use_container_width=True):
                    detail(isbn)


def with_cover(df):
    if "Image-URL-M" in df.columns or "ISBN" not in df.columns:
        return df
    df = df.assign(ISBN=df["ISBN"].astype(str))
    return df.merge(D["bidx"][["Image-URL-M"]].reset_index(), on="ISBN", how="left")


def user_picker(key):
    ss.user = st.selectbox("Pilih pengguna (User-ID)", uids, index=uids.index(ss.user), key=key)
    return ss.user


def style(fig, h=340, legend=False):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      font=dict(family="Plus Jakarta Sans, sans-serif", color=INK, size=13), bargap=.28,
                      margin=dict(l=0, r=16, t=8, b=0), legend=dict(orientation="h", y=1.12, x=0, title=""))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d9cfb4", title="")
    fig.update_yaxes(gridcolor="#efe6d0", zeroline=False, title="")
    return fig


def chart_card(title, fig, note=""):
    with st.container(border=True):
        st.markdown(f'<div class="ct">{title}</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if note:
            st.caption(note)


def hbar(df, x, y, fmt="%{x:,}"):
    d = df.iloc[::-1]
    colors = [GREEN] * (len(d) - 1) + [GOLD]
    fig = go.Figure(go.Bar(x=d[x], y=d[y], orientation="h", marker_color=colors, texttemplate=fmt,
                           textposition="outside", cliponaxis=False))
    fig = style(fig, h=max(300, 34 * len(d)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# =========================================================
# Sidebar
# =========================================================
PAGES = [("Home", "🏠"), ("Rekomendasi", "✨"), ("Segmentasi Pengguna", "👥"),
         ("Eksplorasi Data", "📊"), ("Favorit", "♥"), ("Evaluasi Model", "🎯")]
with st.sidebar:
    st.markdown('<div class="logo">Book Recommender<small>DASHBOARD</small></div>', unsafe_allow_html=True)
    for p, ic in PAGES:
        st.button(f"{ic}  {p}" + (f" ({len(ss.fav)})" if p == "Favorit" and ss.fav else ""), key=f"nav{p}",
                  on_click=go_page, args=(p,), use_container_width=True, type="primary" if ss.page == p else "secondary")

page = ss.page

# =========================================================
# HOME
# =========================================================
if page == "Home":
    with st.container(key="hero"):
        st.markdown('<div class="pill">📖 &nbsp;<b>Book Recommender</b> Dashboard</div>'
                    '<div class="hl"><i>Discover <span class="sp">✦</span></i><span class="m">Your Next</span>'
                    '<i class="g">Favorite Book</i></div><div class="uline"></div>'
                    '<p class="sub">Jelajahi rekomendasi buku yang paling relevan dengan minat dan preferensimu.</p>',
                    unsafe_allow_html=True)
        st.button("Mulai Rekomendasi", key="cta", on_click=go_page, args=("Rekomendasi",))
        st.markdown('<div class="feat"><div><i>📚</i>Hybrid<br>Recommendation</div><div><i>👥</i>User<br>Segmentation</div>'
                    '<div><i>📊</i>Model<br>Evaluation</div></div>', unsafe_allow_html=True)

    heading("Rekomendasi buku untukmu", "pilih pengguna untuk melihat buku yang paling cocok")
    c1, c2 = st.columns([1, 3])
    with c1:
        u = user_picker("home_user")
        st.button("Lihat semua rekomendasi", on_click=go_page, args=("Rekomendasi",))
    recs = recommend(u, float(META.get("best_alpha", 0.1)), 5, int(META.get("like_threshold", 8)))
    if recs.empty:
        st.info("Belum ada rekomendasi untuk pengguna ini.")
    else:
        grid(recs, "me", score="Skor Hybrid")

    q = st.text_input("Cari", placeholder="Cari judul, penulis, atau penerbit...", key="q", label_visibility="collapsed")
    if q.strip():
        m = pd.Series(False, index=D["pop"].index)
        for c in ["Book-Title", "Book-Author", "Publisher"]:
            m |= D["pop"][c].fillna("").astype(str).str.contains(q.strip(), case=False, regex=False)
        res = D["pop"][m].sort_values("n", ascending=False)
        heading("Hasil pencarian", f"{len(res):,} buku")
        grid(res.head(10), "q") if len(res) else st.warning("Tidak ada buku yang cocok. Coba kata kunci lain.")
    else:
        heading("Popular")
        a, b = st.columns([2, 1])
        mode = a.radio("Urutkan", ["Paling banyak dirating", "Rating tertinggi"], horizontal=True, label_visibility="collapsed")
        n_show = b.select_slider("Jumlah", [5, 10, 15], value=5)
        pop = D["pop"]
        pop = pop.sort_values("n", ascending=False) if mode.startswith("Paling") else \
            pop[pop["n"] >= 20].sort_values(["avg", "n"], ascending=False)
        grid(pop.head(n_show), "pop")

    heading("Top Authors", "peringkat berdasarkan jumlah pembaca")
    for rank, (c, (au, r)) in enumerate(zip(st.columns(6), D["au"].head(6).iterrows()), 1):
        ini = "".join(w[0] for w in str(au).split()[:2]).upper()
        with c, st.container(border=True):
            st.markdown(f'<div class="bk"><span class="rk r{rank}">#{rank}</span><div class="av">{html.escape(ini)}</div>'
                        f'<div class="bt" style="min-height:2.4em">{html.escape(str(au)[:26])}</div>'
                        f'<div class="ba">{int(r["Pembaca"]):,} pembaca</div></div>', unsafe_allow_html=True)
            st.button("Lihat buku", key=f"au{rank}", use_container_width=True,
                      on_click=lambda x=au: ss.update(author=None if ss.author == x else x))
    if ss.author:
        heading(f"Buku oleh {ss.author}", "klik tombolnya lagi untuk menutup")
        grid(D["pop"][D["pop"]["Book-Author"] == ss.author].sort_values("n", ascending=False).head(10), "aub")

# =========================================================
# REKOMENDASI HYBRID
# =========================================================
elif page == "Rekomendasi":
    heading("Rekomendasi hybrid", "Collaborative Filtering + Content-Based")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        u = user_picker("rec_user")
    top_n = c2.select_slider("Jumlah buku", [5, 10, 15, 20], value=10)
    alpha = c3.slider("Alpha (bobot CF)", 0.0, 1.0, float(np.clip(META.get("best_alpha", 0.1), 0, 1)), 0.05)
    thr = c4.slider("Ambang buku disukai", 1, 10, int(META.get("like_threshold", 8)))
    recs = recommend(u, alpha, top_n, thr)
    if recs.empty:
        st.warning("Tidak ada buku baru untuk pengguna ini.")
    else:
        grid(recs, "hy", score="Skor Hybrid")
        lab = [f"{i + 1}. {t[:38]}" for i, t in enumerate(recs["Book-Title"])]
        comp = pd.DataFrame({"Buku": lab, "Dari CF": alpha * recs["Skor CF"], "Dari CBF": (1 - alpha) * recs["Skor CBF"]})
        fig = px.bar(comp.melt("Buku", var_name="Sumber", value_name="Skor"), x="Skor", y="Buku", color="Sumber",
                     orientation="h", color_discrete_map={"Dari CF": GD, "Dari CBF": GOLD}, category_orders={"Buku": lab[::-1]})
        chart_card("Dari mana skor tiap rekomendasi berasal", style(fig, h=max(320, 32 * len(recs)), legend=True),
                   f"Bobot CF {alpha:.2f} dan CBF {1 - alpha:.2f}. Semakin panjang bar emas, semakin besar pengaruh kemiripan konten.")
        st.download_button("⬇️ Unduh rekomendasi (CSV)", recs[[c for c in recs.columns if not c.startswith("Image")]]
                           .to_csv(index=False).encode("utf-8-sig"), f"rekomendasi_user_{u}.csv", "text/csv")

# =========================================================
# SEGMENTASI (K-Prototypes)
# =========================================================
elif page == "Segmentasi Pengguna":
    heading("Rekomendasi berdasarkan segmen pengguna", "K-Prototypes · usia dan negara")
    need = ["kproto_final.joblib", "cluster_users.csv", "popular_books.csv"]
    miss = [f for f in need if not (BASE / f).exists()]
    if miss:
        st.warning("Artefak clustering belum ada: " + ", ".join(miss))
    else:
        try:
            kp = joblib.load(BASE / need[0])
            cu, pb = pd.read_csv(BASE / need[1]), pd.read_csv(BASE / need[2])
            ctry = sorted(c for c in cu["Country_Cluster"].dropna().astype(str).unique() if c not in ("Other", "", "nan", "None"))
            c1, c2, c3 = st.columns(3)
            age = c1.slider("Usia", 5, 100, 27)
            country = c2.selectbox("Negara", ctry or ["Other"], index=ctry.index("USA") if "USA" in ctry else 0)
            n_c = c3.select_slider("Jumlah buku", [5, 10, 15, 20], value=10)
            cl = int(kp.predict(np.array([[float(age), country if country in ctry else "Other"]], dtype=object), categorical=[1])[0])
            prof = cu[pd.to_numeric(cu["Cluster"], errors="coerce") == cl]
            cb = pb[pd.to_numeric(pb["Cluster"], errors="coerce") == cl].dropna(subset=["Book-Title"])
            top_c = prof["Country"].mode().iat[0] if not prof.empty else "-"
            k = st.columns(4)
            k[0].metric("Segmen kamu", f"Cluster {cl}")
            k[1].metric("Anggota segmen", f"{len(prof):,}")
            k[2].metric("Rata-rata usia", f"{pd.to_numeric(prof['Age'], errors='coerce').mean():.0f} tahun")
            k[3].metric("Negara dominan", str(top_c))
            if cb.empty:
                st.warning("Belum ada buku untuk segmen ini.")
            else:
                heading("Buku favorit segmen ini", "diurutkan dengan Bayesian Weighted Rating")
                grid(with_cover(cb.sort_values(["Skor_Buku", "Jumlah_Rating"], ascending=False).head(n_c)), "cl", score="Skor_Buku")
        except Exception as exc:
            st.error(f"Gagal memuat model clustering: {exc}")
            st.caption("Periksa versi kmodes dan scikit-learn, serta urutan fitur (Age, Country_Cluster).")

# =========================================================
# EKSPLORASI DATA (ringkasan dari EDA)
# =========================================================
elif page == "Eksplorasi Data":
    heading("Eksplorasi data", "gambaran data yang dipakai model")
    R = RATINGS["Book-Rating"].dropna()
    n_u, n_b = RATINGS["User-ID"].nunique(), RATINGS["ISBN"].nunique()
    k = st.columns(5)
    k[0].metric("Pembaca", f"{n_u:,}")
    k[1].metric("Buku dinilai", f"{n_b:,}")
    k[2].metric("Rating", f"{len(RATINGS):,}")
    k[3].metric("Rata-rata rating", f"{R.mean():.2f}")
    k[4].metric("Matriks kosong", f"{100 * (1 - len(RATINGS) / (n_u * n_b)):.2f}%",
                help="Persentase pasangan pembaca-buku yang belum pernah dinilai.")
    t1, t2, t3 = st.tabs(["Rating & aktivitas", "Pembaca", "Katalog"])

    with t1:
        a, b = st.columns(2)
        dist = R.round().astype(int).value_counts().reindex(range(1, 11), fill_value=0)
        fig = go.Figure(go.Bar(x=dist.index.astype(str), y=dist.values, marker_color=[GOLD if v == dist.max() else GREEN for v in dist.values]))
        with a:
            chart_card("Distribusi rating", style(fig), f"Rating {dist.idxmax()} paling sering diberikan. Pembaca cenderung menilai tinggi.")
        per_u = RATINGS.groupby("User-ID").size()
        bins = pd.cut(per_u, [0, 4, 9, 24, 49, 10**9], labels=["1-4", "5-9", "10-24", "25-49", "50+"]).value_counts().sort_index()
        fig = go.Figure(go.Bar(x=bins.index.astype(str), y=bins.values, marker_color=GREEN))
        with b:
            chart_card("Jumlah rating per pembaca", style(fig), f"Median {per_u.median():.0f} rating per pembaca. Riwayat yang pendek membuat personalisasi sulit.")

    with t2:
        rd = D["readers"]
        if rd is None:
            st.info("Tambahkan cluster_users.csv ke repo untuk menampilkan data pembaca.")
        else:
            a, b = st.columns(2)
            age = pd.to_numeric(rd["Age"], errors="coerce")
            age = age[(age >= 10) & (age <= 90)]
            fig = px.histogram(x=age, color_discrete_sequence=[GREEN])
            fig.update_traces(xbins=dict(start=10, end=90, size=5))
            fig.add_vline(x=age.median(), line_dash="dot", line_color=GOLD, annotation_text=f"median {age.median():.0f}")
            with a:
                chart_card("Usia pembaca", style(fig), f"{len(age):,} dari {len(rd):,} pembaca punya usia valid (10-90 tahun).")
            cc = rd["Country"].dropna().astype(str).str.strip()
            cc = cc[~cc.str.lower().isin(["", "unknown", "nan", "none"])].value_counts().head(10)
            with b:
                chart_card("10 negara asal pembaca terbanyak", hbar(cc.rename_axis("Negara").reset_index(name="Pembaca"), "Pembaca", "Negara"))
            top_r = (RATINGS.groupby("User-ID").agg(Rating=("Book-Rating", "size"), Rata=("Book-Rating", "mean"))
                     .sort_values("Rating", ascending=False).head(10).reset_index().merge(rd, on="User-ID", how="left"))
            top_r.insert(0, "Peringkat", range(1, len(top_r) + 1))
            with st.container(border=True):
                st.markdown('<div class="ct">10 pembaca teraktif</div>', unsafe_allow_html=True)
                st.dataframe(top_r.assign(Rata=top_r["Rata"].round(2)), hide_index=True, use_container_width=True)

    with t3:
        a, b = st.columns(2)
        au = D["au"].head(10).reset_index().rename(columns={"Book-Author": "Penulis"})
        with a:
            chart_card("10 penulis dengan pembaca terbanyak", hbar(au, "Pembaca", "Penulis"), "Dihitung dari jumlah pembaca unik, bukan jumlah rating.")
        yr = pd.to_numeric(BOOKS["Year-Of-Publication"], errors="coerce")
        yr = yr[(yr >= 1950) & (yr <= 2010)]
        dec = (yr // 10 * 10).astype(int).astype(str).add("-an").value_counts().sort_index()
        fig = go.Figure(go.Bar(x=dec.index, y=dec.values, marker_color=GREEN))
        with b:
            chart_card("Tahun terbit buku kandidat", style(fig), "Katalog didominasi buku terbitan 1990-an ke atas.")

# =========================================================
# FAVORIT
# =========================================================
elif page == "Favorit":
    heading("Buku favoritmu", f"{len(ss.fav)} buku")
    if not ss.fav:
        st.info("Belum ada favorit. Tekan **♡ Simpan** pada buku mana pun.")
    else:
        fv = BOOKS.drop_duplicates("ISBN")
        fv = fv[fv["ISBN"].isin(ss.fav)]
        grid(fv, "fv")
        st.button("Kosongkan favorit", on_click=lambda: ss.fav.clear())
        st.download_button("⬇️ Unduh daftar favorit", fv[["ISBN", "Book-Title", "Book-Author"]].to_csv(index=False)
                           .encode("utf-8-sig"), "favorit.csv", "text/csv")

# =========================================================
# EVALUASI
# =========================================================
elif page == "Evaluasi Model":
    heading("Evaluasi model")
    cmp_df, al = D["cmp"], D["alpha"]
    a, b = st.columns(2)
    if not cmp_df.empty:
        ch = cmp_df.melt("Model", value_vars=["RMSE", "MAE"], var_name="Metrik", value_name="Nilai")
        fig = px.bar(ch, x="Model", y="Nilai", color="Metrik", barmode="group", text_auto=".3f",
                     color_discrete_map={"RMSE": GD, "MAE": GOLD})
        best = cmp_df.sort_values("RMSE").iloc[0]
        with a:
            chart_card("Perbandingan model Collaborative Filtering", style(fig, legend=True),
                       f"Terbaik: {best['Model']} (RMSE {best['RMSE']:.3f}). Semakin rendah semakin baik.")
    opts = [c for c in ["Precision@10", "Recall@10", "HitRate@10", "NDCG@10"] if c in al.columns]
    if opts:
        m = st.selectbox("Metrik ranking", opts, index=opts.index("NDCG@10") if "NDCG@10" in opts else 0)
        s = al.sort_values("alpha")
        fig = px.line(s, x="alpha", y=m, markers=True, color_discrete_sequence=[GD])
        bp = s.loc[s[m].idxmax()]
        fig.add_trace(go.Scatter(x=[bp["alpha"]], y=[bp[m]], mode="markers", marker=dict(size=14, color=GOLD, line=dict(width=2, color=GD))))
        with b:
            chart_card(f"{m} pada tiap nilai alpha", style(fig), f"Alpha terbaik untuk {m}: {bp['alpha']:.2f}. Alpha adalah bobot CF; sisanya CBF.")
        with st.expander("Lihat tabel lengkap"):
            st.dataframe(cmp_df, hide_index=True, use_container_width=True)
            st.dataframe(s, hide_index=True, use_container_width=True)
