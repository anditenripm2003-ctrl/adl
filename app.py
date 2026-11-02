import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as sparse
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="BOOK Recommender", page_icon="📚", layout="wide",
                   initial_sidebar_state="expanded")

BASE = Path(__file__).resolve().parent
FILES = {
    "model": "cf_model.joblib", "vectorizer": "tfidf_vectorizer.joblib",
    "tfidf": "tfidf_matrix.npz", "ratings": "model_ratings.pkl",
    "books": "candidate_books.pkl", "cf_comparison": "cf_comparison.csv",
    "alpha_evaluation": "alpha_evaluation.csv", "metadata": "metadata.json",
}
CLUSTER_FILES = ["kproto_final.joblib", "cluster_users.csv", "popular_books.csv"]

# =========================================================
# Gaya: hijau, oranye, kartu putih membulat (mengikuti desain referensi)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
:root{--g:#1fa52e;--gd:#15791f;--o:#ee7a3c;--mint:#e4f8e4;--ink:#2b3a2f;--mu:#8a968d;}
html,body,[class*="css"],[data-testid="stAppViewContainer"]{font-family:'Nunito',sans-serif;color:var(--ink);}
.stApp{background:#f8f8f8;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem;max-width:1500px;}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #eee;}
.logo{font-size:2.1rem;font-weight:800;color:var(--g);letter-spacing:.02em;margin:.2rem 0 1.6rem;}
[data-testid="stSidebar"] button{justify-content:flex-start;border:0;border-radius:40px;
  padding:.65rem 1.1rem;font-weight:700;box-shadow:none;}
[data-testid="stSidebar"] button[kind="secondary"]{background:transparent;color:var(--mu);}
[data-testid="stSidebar"] button[kind="secondary"]:hover{background:var(--mint);color:var(--gd);}
[data-testid="stSidebar"] button[kind="primary"]{background:var(--o);color:#fff;}
.hero{background:linear-gradient(110deg,#1fa52e,#2bb83a);border-radius:22px;padding:2rem 2.4rem;
  color:#fff;display:flex;justify-content:space-between;align-items:center;margin:.8rem 0 1rem;}
.hero h1{font-size:2.1rem;font-weight:800;line-height:1.15;margin:0 0 .6rem;color:#fff;}
.hero p{margin:0;color:#e7ffe9;} .hero .art{font-size:4.6rem;}
.sec{display:flex;gap:1rem;align-items:baseline;margin:1.2rem 0 .4rem;}
.sec b{font-size:1.4rem;font-weight:800;} .sec span{color:var(--g);font-weight:700;}
.bk{text-align:center;} .bt{font-weight:800;font-size:.92rem;line-height:1.2;margin-top:.45rem;min-height:2.2em;}
.ba{color:var(--mu);font-size:.8rem;} .bs{color:var(--gd);font-weight:700;font-size:.82rem;}
.cover{width:100%;max-width:112px;height:150px;object-fit:cover;border-radius:10px;
  box-shadow:0 6px 16px rgba(0,0,0,.18);display:block;margin:0 auto;}
.cover.ph{display:flex;align-items:center;justify-content:center;background:linear-gradient(160deg,#1fa52e,#ee7a3c);
  color:#fff;font-weight:800;font-size:.8rem;padding:.5rem;box-sizing:border-box;}
.av{width:74px;height:74px;border-radius:50%;border:3px solid var(--g);margin:0 auto .3rem;display:flex;
  align-items:center;justify-content:center;font-weight:800;font-size:1.4rem;color:var(--gd);background:var(--mint);}
.tile{background:#fff;border-radius:16px;padding:1rem 1.2rem;box-shadow:0 4px 14px rgba(0,0,0,.05);
  display:flex;gap:1rem;align-items:center;} .tile i{font-style:normal;font-size:1.6rem;background:var(--mint);
  border-radius:12px;padding:.5rem .7rem;} .tile b{font-size:1.5rem;display:block;line-height:1.1;}
.tile small{color:var(--mu);}
.side{background:var(--mint);border-radius:26px;padding:1rem;}
.panel{background:#fff;border-radius:18px;padding:1rem;text-align:center;box-shadow:0 4px 14px rgba(0,0,0,.05);}
.panel .cover{max-width:130px;height:180px;border:3px solid var(--g);} .panel h4{margin:.6rem 0 0;font-weight:800;}
.row{background:#fff;border-radius:12px;padding:.5rem;margin-bottom:.4rem;display:flex;gap:.6rem;align-items:center;}
.row .cover{width:38px;height:54px;max-width:38px;border-radius:6px;margin:0;box-shadow:none;}
.row b{font-size:.85rem;line-height:1.1;display:block;} .row small{color:var(--mu);}
.stButton>button{border-radius:10px;border:1px solid #cfe8d2;background:#fff;color:var(--gd);font-weight:700;
  padding:.2rem .6rem;min-height:0;}
.stButton>button:hover{border-color:var(--g);color:var(--g);}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button{background:var(--o);border-color:var(--o);color:#fff;}
[data-testid="stTextInput"] input{border-radius:30px;background:#eef0f2;border:0;padding:.7rem 1.2rem;}
h2,h3{font-weight:800;}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Muat data
# =========================================================
@st.cache_resource(show_spinner="Memuat model dan data...")
def load():
    miss = [f for f in FILES.values() if not (BASE / f).exists()]
    if miss:
        raise FileNotFoundError("File belum ada: " + ", ".join(miss))
    D = {
        "cf": joblib.load(BASE / FILES["model"]),
        "tfidf": sparse.load_npz(BASE / FILES["tfidf"]),
        "ratings": pd.read_pickle(BASE / FILES["ratings"]),
        "books": pd.read_pickle(BASE / FILES["books"]),
        "cmp": pd.read_csv(BASE / FILES["cf_comparison"]),
        "alpha": pd.read_csv(BASE / FILES["alpha_evaluation"]),
        "meta": json.load(open(BASE / FILES["metadata"], encoding="utf-8")),
    }
    r, b = D["ratings"], D["books"]
    r["User-ID"] = pd.to_numeric(r["User-ID"], errors="coerce")
    r["Book-Rating"] = pd.to_numeric(r["Book-Rating"], errors="coerce")
    r["ISBN"] = r["ISBN"].astype(str)
    b["ISBN"] = b["ISBN"].astype(str)
    D["row"] = {x: i for i, x in enumerate(b["ISBN"])}
    D["bidx"] = b.drop_duplicates("ISBN").set_index("ISBN")
    g = r.groupby("ISBN").agg(n=("Book-Rating", "size"), avg=("Book-Rating", "mean")).reset_index()
    D["stats"] = g.set_index("ISBN")
    D["pop"] = g.merge(b.drop_duplicates("ISBN"), on="ISBN")
    au = D["pop"].groupby("Book-Author")["n"].sum().sort_values(ascending=False)
    D["authors"] = au.head(12).index.tolist()
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
ss.setdefault("sel", None)
ss.setdefault("author", None)


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
    if hist.empty:
        return pd.DataFrame()
    cand = BOOKS[~BOOKS["ISBN"].isin(set(hist["ISBN"]))].copy()
    if cand.empty:
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
    return BOOKS.iloc[idx].assign(Kemiripan=s[idx])


# =========================================================
# Komponen UI
# =========================================================
def cover(url, title=""):
    if isinstance(url, str) and url.startswith("http"):
        return f'<img class="cover" src="{html.escape(url)}" onerror="this.style.display=\'none\'">'
    return f'<div class="cover ph">{html.escape(str(title)[:40])}</div>'


def toggle_fav(i):
    ss.fav.remove(i) if i in ss.fav else ss.fav.append(i)


def pick(i):
    ss.sel = i


def go(p):
    ss.page = p


def grid(df, key, ncols=5, score=None):
    df = df.reset_index(drop=True)
    for s in range(0, len(df), ncols):
        cols = st.columns(ncols)
        for c, (i, row) in zip(cols, df.iloc[s:s + ncols].iterrows()):
            isbn, t = str(row["ISBN"]), str(row.get("Book-Title", ""))
            sub = f'<div class="bs">⭐ {row[score]:.3f}</div>' if score and pd.notna(row.get(score)) else ""
            with c:
                st.markdown(f'<div class="bk">{cover(row.get("Image-URL-M"), t)}<div class="bt">{html.escape(t[:48])}</div>'
                            f'<div class="ba">{html.escape(str(row.get("Book-Author", ""))[:30])}</div>{sub}</div>',
                            unsafe_allow_html=True)
                st.button("♥ Tersimpan" if isbn in ss.fav else "♡ Simpan", key=f"{key}f{i}{isbn}",
                          on_click=toggle_fav, args=(isbn,), use_container_width=True)
                st.button("Detail", key=f"{key}d{i}{isbn}", on_click=pick, args=(isbn,), use_container_width=True)


def book_info(isbn):
    if isbn not in D["bidx"].index:
        return None
    d = D["bidx"].loc[isbn].to_dict()
    d["ISBN"] = isbn
    return d


def with_cover(df):
    """Tambahkan cover/ISBN dari katalog untuk data hasil cluster."""
    if "Image-URL-M" in df.columns or "ISBN" not in df.columns:
        return df
    df = df.copy()
    df["ISBN"] = df["ISBN"].astype(str)
    return df.merge(D["bidx"][["Image-URL-M"]].reset_index(), on="ISBN", how="left")


def heading(t, extra=""):
    st.markdown(f'<div class="sec"><b>{t}</b><span>{extra}</span></div>', unsafe_allow_html=True)


# =========================================================
# Sidebar navigasi
# =========================================================
PAGES = [("Home", "🏠"), ("Rekomendasi Hybrid", "✨"), ("Rekomendasi Cluster", "👥"),
         ("Favorit", "♥"), ("Evaluasi Model", "📊"), ("Data Buku", "📖")]
with st.sidebar:
    st.markdown('<div class="logo">BOOK</div>', unsafe_allow_html=True)
    for p, ic in PAGES:
        label = f"{ic}  {p}" + (f" ({len(ss.fav)})" if p == "Favorit" and ss.fav else "")
        st.button(label, key=f"nav{p}", on_click=go, args=(p,), use_container_width=True,
                  type="primary" if ss.page == p else "secondary")

main, side = st.columns([3.1, 1.2], gap="large")

# =========================================================
# Panel kanan: pengguna, detail buku, buku berikutnya, target
# =========================================================
with side:
    st.markdown('<div class="side">', unsafe_allow_html=True)
    uids = sorted(RATINGS["User-ID"].dropna().astype(int).unique().tolist())
    most = int(RATINGS.groupby("User-ID").size().idxmax())
    uid = st.selectbox("Pengguna", uids, index=uids.index(most), key="uid")

    heading("Detail buku")
    info = book_info(ss.sel) if ss.sel else None
    if info:
        stt = D["stats"].loc[ss.sel] if ss.sel in D["stats"].index else None
        meta_line = f"{info.get('Year-Of-Publication', '')} · {info.get('Publisher', '')}"
        rate = f"⭐ {stt['avg']:.1f} dari {int(stt['n'])} rating" if stt is not None else ""
        st.markdown(f'<div class="panel">{cover(info.get("Image-URL-L") or info.get("Image-URL-M"), info.get("Book-Title"))}'
                    f'<h4>{html.escape(str(info.get("Book-Title", "")))}</h4>'
                    f'<div class="ba">{html.escape(str(info.get("Book-Author", "")))}</div>'
                    f'<div class="ba">{html.escape(meta_line)}</div><div class="bs">{rate}</div></div>',
                    unsafe_allow_html=True)
        st.button("♥ Tersimpan" if ss.sel in ss.fav else "♡ Simpan ke favorit", key="sf",
                  on_click=toggle_fav, args=(ss.sel,), use_container_width=True)
        heading("Buku mirip")
        for i, r in similar(ss.sel).iterrows():
            st.markdown(f'<div class="row">{cover(r.get("Image-URL-S") or r.get("Image-URL-M"), "")}'
                        f'<div><b>{html.escape(str(r["Book-Title"])[:40])}</b>'
                        f'<small>{html.escape(str(r["Book-Author"])[:26])}</small></div></div>', unsafe_allow_html=True)
            st.button("Lihat", key=f"sim{i}{r['ISBN']}", on_click=pick, args=(str(r["ISBN"]),))
    else:
        st.info("Klik **Detail** pada sebuah buku untuk melihat info dan buku yang mirip.")

    heading("Next books", f"untuk User {uid}")
    nxt = recommend(uid, float(META.get("best_alpha", 0.1)), 3, int(META.get("like_threshold", 8)))
    for i, r in nxt.iterrows():
        st.markdown(f'<div class="row">{cover(r.get("Image-URL-S") or r.get("Image-URL-M"), "")}'
                    f'<div><b>{html.escape(str(r["Book-Title"])[:40])}</b>'
                    f'<small>{html.escape(str(r["Book-Author"])[:26])}</small></div></div>', unsafe_allow_html=True)
        st.button("Detail", key=f"nx{i}{r['ISBN']}", on_click=pick, args=(str(r["ISBN"]),))

    heading("Target baca")
    goal = st.slider("Target buku favorit", 1, 10, 5, key="goal")
    st.progress(min(len(ss.fav) / goal, 1.0))
    st.caption(f"{len(ss.fav)}/{goal} buku disimpan")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Konten utama
# =========================================================
with main:
    q = st.text_input("Cari", placeholder="Search your book...", key="q", label_visibility="collapsed")
    page = ss.page

    # ---------------- HOME ----------------
    if page == "Home":
        if q.strip():
            m = pd.Series(False, index=D["pop"].index)
            for c in ["Book-Title", "Book-Author", "ISBN", "Publisher"]:
                if c in D["pop"]:
                    m |= D["pop"][c].fillna("").astype(str).str.contains(q.strip(), case=False, regex=False)
            res = D["pop"][m].sort_values("n", ascending=False)
            heading("Hasil pencarian", f"{len(res):,} buku")
            if res.empty:
                st.warning("Tidak ada buku yang cocok. Coba kata kunci lain.")
            else:
                grid(res.head(10), "q")
        else:
            st.markdown('<div class="hero"><div><h1>TRENDING BOOKS<br>THIS MONTH</h1>'
                        '<p>Temukan buku yang paling banyak dinilai dan rekomendasi khusus untukmu.</p></div>'
                        '<div class="art">📚🧑‍🎓</div></div>', unsafe_allow_html=True)
            st.button("LIHAT REKOMENDASI UNTUKKU", type="primary", on_click=go, args=("Rekomendasi Hybrid",))

            heading("Popular", "")
            a, b = st.columns([2, 1])
            mode = a.radio("Urutkan", ["Paling banyak dirating", "Rating tertinggi"], horizontal=True,
                           label_visibility="collapsed")
            n_show = b.select_slider("Jumlah", [5, 10, 15], value=5)
            pop = D["pop"]
            pop = pop.sort_values("n", ascending=False) if mode.startswith("Paling") else \
                pop[pop["n"] >= 20].sort_values(["avg", "n"], ascending=False)
            grid(pop.head(n_show), "pop")

            heading("Top Authors")
            cols = st.columns(6)
            for c, au in zip(cols, D["authors"][:6]):
                ini = "".join(w[0] for w in str(au).split()[:2]).upper()
                with c:
                    st.markdown(f'<div class="av">{html.escape(ini)}</div>', unsafe_allow_html=True)
                    st.button(str(au)[:16], key=f"au{au}", use_container_width=True,
                              on_click=lambda x=au: ss.update(author=None if ss.author == x else x))
            if ss.author:
                heading(f"Buku oleh {ss.author}", "klik nama lagi untuk menutup")
                grid(D["pop"][D["pop"]["Book-Author"] == ss.author].sort_values("n", ascending=False).head(10),
                     "aub")

            st.write("")
            t = st.columns(3)
            for c, (ic, v, lb) in zip(t, [("📘", f"{RATINGS['User-ID'].nunique():,}", "Pengguna"),
                                           ("🧑‍🤝‍🧑", f"{BOOKS['ISBN'].nunique():,}", "Buku kandidat"),
                                           ("📖", f"{len(RATINGS):,}", "Rating")]):
                c.markdown(f'<div class="tile"><i>{ic}</i><div><b>{v}</b><small>{lb}</small></div></div>',
                           unsafe_allow_html=True)

    # ---------------- HYBRID ----------------
    elif page == "Rekomendasi Hybrid":
        heading("Rekomendasi hybrid", "CF + Content-Based")
        c1, c2, c3 = st.columns(3)
        top_n = c1.select_slider("Jumlah buku", [5, 10, 15, 20, 30], value=10)
        alpha = c2.slider("Alpha (bobot CF)", 0.0, 1.0, float(np.clip(META.get("best_alpha", 0.1), 0, 1)), 0.05)
        thr = c3.slider("Ambang buku disukai", 1, 10, int(META.get("like_threshold", 8)))
        st.caption(f"Pengguna dipilih di panel kanan: User {uid} · CF {alpha:.2f} · CBF {1 - alpha:.2f}")
        recs = recommend(uid, alpha, top_n, thr)
        if recs.empty:
            st.warning("Tidak ada buku baru untuk pengguna ini.")
        else:
            grid(recs, "hy", score="Skor Hybrid")
            fig = px.bar(recs.sort_values("Skor Hybrid"), x="Skor Hybrid", y="Book-Title", orientation="h",
                         hover_data=["Skor CF", "Skor CBF", "Prediksi CF"], color_discrete_sequence=["#1fa52e"])
            fig.update_layout(height=max(380, 30 * len(recs)), yaxis_title="", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Unduh CSV", recs.drop(columns=[c for c in recs if c.startswith("Image")])
                               .to_csv(index=False).encode("utf-8-sig"), f"rekomendasi_user_{uid}.csv", "text/csv")

    # ---------------- CLUSTER ----------------
    elif page == "Rekomendasi Cluster":
        heading("Rekomendasi berdasarkan cluster pengguna", "K-Prototypes")
        miss = [f for f in CLUSTER_FILES if not (BASE / f).exists()]
        if miss:
            st.warning("Artefak clustering belum ada: " + ", ".join(miss))
        else:
            try:
                kp = joblib.load(BASE / "kproto_final.joblib")
                cu = pd.read_csv(BASE / "cluster_users.csv")
                pb = pd.read_csv(BASE / "popular_books.csv")
                ctry = sorted(c for c in cu["Country_Cluster"].dropna().astype(str).unique()
                              if c not in ("Other", "", "nan", "None"))
                c1, c2, c3 = st.columns(3)
                age = c1.slider("Usia", 5, 100, 27)
                country = c2.selectbox("Negara", ctry or ["Other"],
                                       index=ctry.index("USA") if "USA" in ctry else 0)
                n_c = c3.select_slider("Jumlah buku", [5, 10, 15, 20], value=10)
                cl = int(kp.predict(np.array([[float(age), country if country in ctry else "Other"]],
                                             dtype=object), categorical=[1])[0])
                prof = cu[pd.to_numeric(cu["Cluster"], errors="coerce") == cl]
                cb = pb[pd.to_numeric(pb["Cluster"], errors="coerce") == cl].dropna(subset=["Book-Title"])
                k = st.columns(3)
                k[0].metric("Cluster diprediksi", f"Cluster {cl}")
                k[1].metric("Pengguna di cluster", f"{len(prof):,}")
                k[2].metric("Rata-rata usia", f"{pd.to_numeric(prof['Age'], errors='coerce').mean():.1f}")
                if cb.empty:
                    st.warning("Belum ada buku untuk cluster ini.")
                else:
                    cb = with_cover(cb.sort_values(["Skor_Buku", "Jumlah_Rating"], ascending=False).head(n_c))
                    grid(cb, "cl", score="Skor_Buku")
                    fig = px.bar(cb.sort_values("Skor_Buku"), x="Skor_Buku", y="Book-Title", orientation="h",
                                 hover_data=["Jumlah_Rating", "Rata_rata_Rating"], color_discrete_sequence=["#ee7a3c"])
                    fig.update_layout(height=max(380, 30 * len(cb)), yaxis_title="", margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Gagal memuat model clustering: {exc}")
                st.caption("Periksa versi kmodes/scikit-learn dan urutan fitur (Age, Country_Cluster).")

    # ---------------- FAVORIT ----------------
    elif page == "Favorit":
        heading("Buku favoritmu", f"{len(ss.fav)} buku")
        if not ss.fav:
            st.info("Belum ada favorit. Tekan **♡ Simpan** pada buku mana pun.")
        else:
            fv = BOOKS.drop_duplicates("ISBN")
            fv = fv[fv["ISBN"].isin(ss.fav)]
            grid(fv, "fv")
            st.button("Kosongkan favorit", on_click=lambda: ss.fav.clear())
            st.download_button("⬇️ Unduh daftar favorit", fv[["ISBN", "Book-Title", "Book-Author"]]
                               .to_csv(index=False).encode("utf-8-sig"), "favorit.csv", "text/csv")

    # ---------------- EVALUASI ----------------
    elif page == "Evaluasi Model":
        heading("Evaluasi model")
        cmp_df = D["cmp"]
        if not cmp_df.empty:
            ch = cmp_df.melt(id_vars="Model", value_vars=["RMSE", "MAE"], var_name="Metrik", value_name="Nilai")
            st.plotly_chart(px.bar(ch, x="Model", y="Nilai", color="Metrik", barmode="group", text_auto=".3f",
                                   color_discrete_sequence=["#1fa52e", "#ee7a3c"]), use_container_width=True)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        al = D["alpha"]
        opts = [c for c in ["Precision@10", "Recall@10", "HitRate@10", "NDCG@10"] if c in al.columns]
        if opts:
            m = st.selectbox("Metrik ranking", opts, index=opts.index("NDCG@10") if "NDCG@10" in opts else 0)
            st.plotly_chart(px.line(al.sort_values("alpha"), x="alpha", y=m, markers=True,
                                    color_discrete_sequence=["#1fa52e"]), use_container_width=True)
            st.dataframe(al.sort_values("alpha"), use_container_width=True, hide_index=True)

    # ---------------- DATA ----------------
    elif page == "Data Buku":
        heading("Data buku & rating")
        cnt = RATINGS["Book-Rating"].value_counts().sort_index().rename_axis("Rating").reset_index(name="Jumlah")
        st.plotly_chart(px.bar(cnt, x="Rating", y="Jumlah", color_discrete_sequence=["#1fa52e"]),
                        use_container_width=True)
        view = BOOKS[[c for c in ["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]
                      if c in BOOKS.columns]]
        if q.strip():
            mk = pd.Series(False, index=view.index)
            for c in view.columns:
                mk |= view[c].fillna("").astype(str).str.contains(q.strip(), case=False, regex=False)
            view = view[mk]
        st.dataframe(view.head(500), use_container_width=True, hide_index=True)
        st.caption(f"Menampilkan maksimal 500 dari {len(view):,} baris. Gunakan kotak pencarian di atas untuk memfilter.")
