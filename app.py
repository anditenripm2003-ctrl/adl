import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Konfigurasi
# =========================================================
st.set_page_config(
    page_title="Book Recommendation Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = {
    "model": BASE_DIR / "cf_model.joblib",
    "vectorizer": BASE_DIR / "tfidf_vectorizer.joblib",
    "tfidf": BASE_DIR / "tfidf_matrix.npz",
    "ratings": BASE_DIR / "model_ratings.pkl",
    "books": BASE_DIR / "candidate_books.pkl",
    "cf_comparison": BASE_DIR / "cf_comparison.csv",
    "alpha_evaluation": BASE_DIR / "alpha_evaluation.csv",
    "metadata": BASE_DIR / "metadata.json",
    # Artefak clustering (diekspor dari notebook K-Prototypes)
    "cluster_model": BASE_DIR / "kproto_final.joblib",
    "cluster_users": BASE_DIR / "cluster_users.csv",
    "popular_books": BASE_DIR / "popular_books.csv",
}

# =========================================================
# Gaya dashboard
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --green-dark: #173f2b;
        --green: #247447;
        --green-mid: #3f8f55;
        --green-soft: #eaf5e9;
        --yellow: #f2d34f;
        --yellow-soft: #fff8d9;
        --ink: #20352a;
        --muted: #68796d;
        --surface: #ffffff;
        --line: #e2ebdf;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    h1, h2, h3, h4, [data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.035em;
    }
    .stApp {background: #f6f9f2;}
    [data-testid="stHeader"] {background: rgba(246,249,242,.88);}
    .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1450px;}

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        background: linear-gradient(115deg, #173f2b 0%, #247447 68%, #3f8f55 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(23,63,43,.13);
        border: 1px solid rgba(255,255,255,.14);
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 180px; height: 180px;
        border-radius: 50%;
        right: -35px; top: -70px;
        background: rgba(242,211,79,.22);
    }
    .hero h1 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: clamp(1.8rem, 3vw, 2.45rem);
        font-weight: 800; line-height: 1.2;
        margin: 0 0 .55rem 0; color: #ffffff;
    }
    .hero p {font-size: 1rem; color: #edf8e9; margin: 0; line-height: 1.65;}

    [data-testid="stSidebar"] {
        background: #edf5e8;
        border-right: 1px solid #dce9d7;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {color: var(--green-dark);}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {font-weight: 600;}

    div[data-testid="stMetric"] {
        background: var(--surface);
        padding: 1.1rem 1.15rem;
        border-radius: 17px;
        border: 1px solid var(--line);
        box-shadow: 0 5px 18px rgba(32,53,42,.035);
    }
    [data-testid="stMetricLabel"] {color: var(--muted); font-weight: 600;}
    [data-testid="stMetricValue"] {color: var(--green-dark); font-weight: 800;}

    h2, h3 {color: var(--green-dark);}
    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 11px;
        border: 1px solid var(--green);
        background: var(--green);
        color: #ffffff;
        font-weight: 700;
        padding: .55rem 1rem;
        transition: all .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: var(--green-dark);
        border-color: var(--green-dark);
        color: #ffffff;
        transform: translateY(-1px);
    }
    div[role="radiogroup"] label, [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label, [data-testid="stTextInput"] label {
        color: var(--green-dark);
        font-weight: 600;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 13px;
        overflow: hidden;
    }
    .small-note {color: var(--muted); font-size: .85rem;}
    hr {border-color: var(--line);}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Muat artefak
# =========================================================
@st.cache_resource(show_spinner="Memuat model dan data rekomendasi...")
def load_artifacts():
    missing = [name for name, path in ARTIFACTS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Artefak berikut belum ditemukan: " + ", ".join(missing)
            + ". Letakkan semua file artefak dalam folder yang sama dengan app.py."
        )

    model = joblib.load(ARTIFACTS["model"])
    vectorizer = joblib.load(ARTIFACTS["vectorizer"])
    tfidf = sparse.load_npz(ARTIFACTS["tfidf"])
    ratings = pd.read_pickle(ARTIFACTS["ratings"])
    books = pd.read_pickle(ARTIFACTS["books"])
    cf_comparison = pd.read_csv(ARTIFACTS["cf_comparison"])
    alpha_eval = pd.read_csv(ARTIFACTS["alpha_evaluation"])
    with open(ARTIFACTS["metadata"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Pastikan format kolom inti konsisten
    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors="coerce")
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings["ISBN"] = ratings["ISBN"].astype(str)
    books["ISBN"] = books["ISBN"].astype(str)
    if "Year-Of-Publication" in books:
        books["Year-Of-Publication"] = pd.to_numeric(
            books["Year-Of-Publication"], errors="coerce"
        )

    isbn_to_row = {isbn: i for i, isbn in enumerate(books["ISBN"].astype(str))}
    return model, vectorizer, tfidf, ratings, books, cf_comparison, alpha_eval, metadata, isbn_to_row

try:
    (cf_model, vectorizer, tfidf_matrix, model_ratings, candidate_books,
     cf_comparison, alpha_evaluation, metadata, isbn_to_row) = load_artifacts()
except Exception as exc:
    st.error(f"Dashboard belum dapat dimuat: {exc}")
    st.info(
        "Pastikan semua file artefak hasil training tersedia bersama app.py. "
        "Lihat README.txt untuk instruksi menjalankan aplikasi."
    )
    st.stop()

# =========================================================
# Fungsi rekomendasi hybrid (mengikuti notebook)
# alpha = bobot Collaborative Filtering; 1-alpha = bobot CBF
# =========================================================
def build_user_profile(user_history, like_threshold):
    hist = user_history[
        user_history["ISBN"].astype(str).isin(isbn_to_row)
    ].copy()
    if hist.empty:
        return None

    liked = hist[hist["Book-Rating"] >= like_threshold].copy()
    if liked.empty:
        liked = hist.nlargest(min(5, len(hist)), "Book-Rating").copy()

    rows = [isbn_to_row[str(isbn)] for isbn in liked["ISBN"]]
    weights = liked["Book-Rating"].to_numpy(dtype=float) / 10.0
    selected_vectors = tfidf_matrix[rows]
    weighted_vectors = selected_vectors.multiply(weights[:, None])
    profile = sparse.csr_matrix(
        weighted_vectors.sum(axis=0) / max(weights.sum(), 1e-9)
    )
    return profile

def recommend_for_user(user_id, alpha, top_n, like_threshold):
    history = model_ratings[model_ratings["User-ID"] == int(user_id)].copy()
    if history.empty:
        raise ValueError("User tidak ditemukan pada data model.")

    seen = set(history["ISBN"].astype(str))
    candidates = candidate_books[
        ~candidate_books["ISBN"].astype(str).isin(seen)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    isbns = candidates["ISBN"].astype(str).tolist()
    estimates = np.array([
        cf_model.predict(int(user_id), str(isbn)).est for isbn in isbns
    ], dtype=float)
    cf_scores = np.clip((estimates - 1.0) / 9.0, 0.0, 1.0)

    profile = build_user_profile(history, like_threshold)
    rows = [isbn_to_row[x] for x in isbns]
    if profile is None:
        cbf_scores = np.zeros(len(rows))
    else:
        cbf_scores = cosine_similarity(profile, tfidf_matrix[rows]).ravel()

    hybrid_scores = float(alpha) * cf_scores + (1.0 - float(alpha)) * cbf_scores

    columns = [
        c for c in [
            "ISBN", "Book-Title", "Book-Author", "Publisher",
            "Year-Of-Publication", "Image-URL-M", "Image-URL-L"
        ] if c in candidates.columns
    ]
    result = candidates[columns].copy()
    result["Prediksi Rating CF"] = estimates
    result["Skor CF"] = cf_scores
    result["Skor CBF"] = cbf_scores
    result["Skor Hybrid"] = hybrid_scores
    result = result.sort_values("Skor Hybrid", ascending=False).head(top_n).reset_index(drop=True)
    result.index = result.index + 1
    result.index.name = "Rank"
    return result

# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "Ringkasan",
        "Rekomendasi Buku",
        "Rekomendasi Berdasarkan Cluster",
        "Evaluasi Model",
        "Data Buku & Rating",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Hybrid Recommender System")
st.sidebar.caption("Collaborative Filtering + Content-Based Filtering")

# =========================================================
# Header
# =========================================================
st.markdown("""
<div class="hero">
  <h1>📚 Book Recommendation Dashboard</h1>
  <p>Sistem rekomendasi buku hybrid menggunakan Collaborative Filtering dan Content-Based Filtering.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Halaman ringkasan
# =========================================================
if page == "Ringkasan":
    st.subheader("Ringkasan Sistem")
    n_users = int(model_ratings["User-ID"].nunique())
    n_books = int(candidate_books["ISBN"].nunique())
    n_ratings = int(len(model_ratings))
    best_alpha = float(metadata.get("best_alpha", 0.1))
    cf_name = metadata.get("selected_cf_model", "Tidak diketahui")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pengguna pada model", f"{n_users:,}")
    c2.metric("Buku kandidat", f"{n_books:,}")
    c3.metric("Rating eksplisit", f"{n_ratings:,}")
    c4.metric("Model CF terpilih", cf_name)

    st.markdown("### Cara kerja sistem")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 1. Collaborative Filtering")
        st.write("Memprediksi rating berdasarkan pola rating pengguna lain. Model terpilih dan parameternya dimuat dari artefak training.")
    with b:
        st.markdown("#### 2. Content-Based Filtering")
        st.write("Menghitung kemiripan konten buku berdasarkan judul, penulis, penerbit, dan tahun publikasi menggunakan TF-IDF.")
    with c:
        st.markdown("#### 3. Skor Hybrid")
        st.write("Menggabungkan skor CF dan CBF untuk mengurutkan buku yang belum pernah dinilai pengguna.")

    st.markdown("### Bobot hybrid")
    m1, m2, m3 = st.columns(3)
    m1.metric("Alpha / bobot CF", f"{best_alpha:.2f}")
    m2.metric("Bobot CBF", f"{1-best_alpha:.2f}")
    m3.metric("Ambang buku disukai", f"Rating ≥ {metadata.get('like_threshold', 8)}")

    st.markdown("### Perbandingan model Collaborative Filtering")
    if not cf_comparison.empty:
        chart = cf_comparison.melt(
            id_vars="Model", value_vars=["RMSE", "MAE"],
            var_name="Metrik", value_name="Nilai"
        )
        fig = px.bar(
            chart, x="Model", y="Nilai", color="Metrik", barmode="group",
            title="Perbandingan RMSE dan MAE (lebih rendah lebih baik)",
            text_auto=".3f"
        )
        fig.update_layout(legend_title_text="", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cf_comparison, use_container_width=True, hide_index=True)
    st.caption("Catatan: angka pada dashboard dibaca dari artefak evaluasi hasil notebook.")

# =========================================================
# Halaman rekomendasi
# =========================================================
elif page == "Rekomendasi Buku":
    st.subheader("Temukan rekomendasi buku")
    user_ids = sorted(model_ratings["User-ID"].dropna().astype(int).unique().tolist())
    if not user_ids:
        st.warning("Tidak ada User-ID yang tersedia.")
        st.stop()

    most_active = int(model_ratings.groupby("User-ID").size().idxmax())
    with st.form("recommendation_form"):
        left, right = st.columns([1, 1])
        with left:
            selected_user = st.selectbox(
                "Pilih User-ID",
                options=user_ids,
                index=user_ids.index(most_active) if most_active in user_ids else 0,
                help="Daftar pengguna yang tersedia pada data model."
            )
            top_n = st.slider("Jumlah rekomendasi", min_value=5, max_value=30, value=10, step=5)
        with right:
            default_alpha = float(metadata.get("best_alpha", 0.1))
            alpha = st.slider(
                "Alpha — bobot Collaborative Filtering",
                min_value=0.0, max_value=1.0,
                value=float(np.clip(default_alpha, 0, 1)), step=0.05,
                help="Bobot CBF otomatis = 1 − alpha."
            )
            like_threshold = st.slider(
                "Ambang rating buku disukai", 1, 10,
                int(metadata.get("like_threshold", 8))
            )
        submitted = st.form_submit_button("✨ Tampilkan Rekomendasi", type="primary")

    if submitted:
        try:
            recs = recommend_for_user(selected_user, alpha, top_n, like_threshold)
            if recs.empty:
                st.warning("Tidak ada buku baru untuk direkomendasikan kepada pengguna ini.")
            else:
                st.success(f"Berikut {len(recs)} rekomendasi untuk User-ID {selected_user}.")
                st.caption(f"Bobot yang digunakan: CF {alpha:.2f} · CBF {1-alpha:.2f}")
                score_chart = recs.reset_index().sort_values("Skor Hybrid", ascending=True)
                fig = px.bar(
                    score_chart, x="Skor Hybrid", y="Book-Title", orientation="h",
                    hover_data=["Book-Author", "Skor CF", "Skor CBF", "Prediksi Rating CF"],
                    title="Peringkat rekomendasi berdasarkan skor hybrid",
                    text_auto=".3f"
                )
                fig.update_layout(height=max(420, 32 * len(recs)), yaxis_title="", xaxis_title="Skor Hybrid")
                st.plotly_chart(fig, use_container_width=True)

                display_cols = [
                    c for c in [
                        "ISBN", "Book-Title", "Book-Author", "Publisher",
                        "Year-Of-Publication", "Prediksi Rating CF",
                        "Skor CF", "Skor CBF", "Skor Hybrid"
                    ] if c in recs.columns
                ]
                shown = recs[display_cols].copy()
                for col in ["Prediksi Rating CF", "Skor CF", "Skor CBF", "Skor Hybrid"]:
                    if col in shown:
                        shown[col] = shown[col].round(4)
                st.dataframe(shown, use_container_width=True)
                st.download_button(
                    "⬇️ Unduh hasil rekomendasi (CSV)",
                    data=recs.to_csv(index=True).encode("utf-8-sig"),
                    file_name=f"rekomendasi_user_{selected_user}.csv",
                    mime="text/csv"
                )

                if "Image-URL-M" in recs.columns:
                    st.markdown("### Pratinjau buku")
                    image_rows = recs.head(min(6, len(recs)))
                    cards = st.columns(3)
                    for i, (_, row) in enumerate(image_rows.iterrows()):
                        with cards[i % 3]:
                            image_url = row.get("Image-URL-M", "")
                            if pd.notna(image_url) and str(image_url).startswith("http"):
                                st.image(str(image_url), width=125)
                            st.markdown(f"**{row.get('Book-Title', 'Judul tidak tersedia')}**")
                            st.caption(str(row.get("Book-Author", "Penulis tidak diketahui")))
                            st.caption(f"Skor hybrid: {row.get('Skor Hybrid', 0):.3f}")
        except Exception as exc:
            st.error(f"Gagal membuat rekomendasi: {exc}")

# =========================================================
# Rekomendasi berbasis clustering K-Prototypes
# =========================================================
elif page == "Rekomendasi Berdasarkan Cluster":
    st.subheader("Rekomendasi Buku Berdasarkan Cluster Pengguna")
    st.write(
        "Masukkan usia dan negara untuk memprediksi cluster pengguna. "
        "Buku kemudian diurutkan berdasarkan Bayesian Weighted Rating "
        "yang dihitung untuk cluster tersebut."
    )

    required_cluster_files = [
        ARTIFACTS["cluster_model"],
        ARTIFACTS["cluster_users"],
        ARTIFACTS["popular_books"],
    ]
    missing_cluster_files = [p.name for p in required_cluster_files if not p.exists()]

    if missing_cluster_files:
        st.warning(
            "Artefak clustering belum ditemukan. Ekspor tiga file berikut dari notebook "
            "dan letakkan di folder yang sama dengan app.py:"
        )
        for filename in [
            "kproto_final.joblib",
            "cluster_users.csv",
            "popular_books.csv",
        ]:
            st.code(filename, language=None)
        st.info(
            "Di bagian akhir jawaban, gunakan kode ekspor yang disediakan untuk "
            "membuat ketiga file tersebut. Halaman rekomendasi hybrid tetap dapat digunakan."
        )
    else:
        try:
            cluster_model = joblib.load(ARTIFACTS["cluster_model"])
            cluster_users_df = pd.read_csv(ARTIFACTS["cluster_users"])
            popular_books_df = pd.read_csv(ARTIFACTS["popular_books"])

            required_user_cols = {"Age", "Country", "Country_Cluster", "Cluster"}
            required_book_cols = {
                "Cluster", "Book-Title", "Book-Author",
                "Jumlah_Rating", "Rata_rata_Rating", "Skor_Buku"
            }
            if not required_user_cols.issubset(cluster_users_df.columns):
                raise ValueError(
                    "Kolom cluster_users.csv tidak lengkap. "
                    f"Kolom yang dibutuhkan: {sorted(required_user_cols)}"
                )
            if not required_book_cols.issubset(popular_books_df.columns):
                raise ValueError(
                    "Kolom popular_books.csv tidak lengkap. "
                    f"Kolom yang dibutuhkan: {sorted(required_book_cols)}"
                )

            valid_countries = sorted(
                cluster_users_df.loc[
                    cluster_users_df["Country_Cluster"].astype(str) != "Other",
                    "Country_Cluster"
                ].dropna().astype(str).unique().tolist()
            )
            valid_countries = [c for c in valid_countries if c not in ["", "nan", "None"]]

            with st.form("cluster_recommendation_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age_value = st.number_input(
                        "Usia pengguna",
                        min_value=5,
                        max_value=100,
                        value=27,
                        step=1,
                    )
                    n_cluster_books = st.slider(
                        "Jumlah rekomendasi buku",
                        min_value=5,
                        max_value=20,
                        value=10,
                        step=5,
                    )
                with col2:
                    country_options = valid_countries if valid_countries else ["Other"]
                    default_country_index = (
                        country_options.index("USA") if "USA" in country_options else 0
                    )
                    country_value = st.selectbox(
                        "Negara pengguna",
                        options=country_options,
                        index=default_country_index,
                    )
                cluster_submit = st.form_submit_button(
                    "🔎 Prediksi Cluster & Rekomendasikan Buku",
                    type="primary",
                )

            if cluster_submit:
                # Ikuti urutan fitur saat training: Age, Country_Cluster.
                country_for_model = (
                    country_value if country_value in valid_countries else "Other"
                )
                new_user = np.array(
                    [[float(age_value), str(country_for_model)]],
                    dtype=object,
                )
                predicted_cluster = int(
                    cluster_model.predict(new_user, categorical=[1])[0]
                )

                cluster_profile = cluster_users_df[
                    pd.to_numeric(cluster_users_df["Cluster"], errors="coerce")
                    == predicted_cluster
                ]
                cluster_books = popular_books_df[
                    pd.to_numeric(popular_books_df["Cluster"], errors="coerce")
                    == predicted_cluster
                ].copy()

                k1, k2, k3 = st.columns(3)
                k1.metric("Cluster diprediksi", f"Cluster {predicted_cluster}")
                k2.metric("Pengguna dalam cluster", f"{len(cluster_profile):,}")
                k3.metric("Buku tersedia", f"{cluster_books['ISBN'].nunique() if 'ISBN' in cluster_books.columns else len(cluster_books):,}")

                if not cluster_profile.empty:
                    avg_age = pd.to_numeric(
                        cluster_profile["Age"], errors="coerce"
                    ).mean()
                    st.caption(
                        f"Profil cluster: rata-rata usia sekitar {avg_age:.1f} tahun. "
                        "Karakteristik ini merupakan ringkasan data training, bukan batas mutlak cluster."
                    )

                if cluster_books.empty:
                    st.warning("Belum ada rekomendasi buku yang tersedia untuk cluster ini.")
                else:
                    cluster_books = cluster_books.dropna(subset=["Book-Title"])
                    cluster_books = cluster_books.sort_values(
                        ["Skor_Buku", "Jumlah_Rating"],
                        ascending=[False, False],
                    ).head(n_cluster_books)

                    st.markdown("### Buku yang direkomendasikan")
                    chart_df = cluster_books.copy()
                    chart_df["Label Buku"] = chart_df["Book-Title"].astype(str).str.slice(0, 55)
                    fig = px.bar(
                        chart_df.sort_values("Skor_Buku", ascending=True),
                        x="Skor_Buku",
                        y="Label Buku",
                        orientation="h",
                        hover_data=[
                            c for c in [
                                "Book-Title", "Book-Author", "Jumlah_Rating",
                                "Rata_rata_Rating"
                            ] if c in chart_df.columns
                        ],
                        title="Peringkat buku berdasarkan Bayesian Weighted Rating",
                        text_auto=".3f",
                    )
                    fig.update_layout(
                        height=max(420, 34 * len(chart_df)),
                        yaxis_title="",
                        xaxis_title="Skor Bayesian Weighted Rating",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    display_columns = [
                        c for c in [
                            "ISBN", "Book-Title", "Book-Author", "Jumlah_Rating",
                            "Rata_rata_Rating", "Rata_rata_Cluster", "Skor_Buku"
                        ] if c in cluster_books.columns
                    ]
                    st.dataframe(
                        cluster_books[display_columns],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "⬇️ Unduh rekomendasi cluster (CSV)",
                        data=cluster_books[display_columns].to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"rekomendasi_cluster_{predicted_cluster}.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Gagal memuat model clustering: {exc}")
            st.caption(
                "Pastikan model K-Prototypes, versi pustaka, nama kolom, dan urutan fitur "
                "sama dengan yang digunakan saat training di notebook."
            )

# =========================================================
# Halaman evaluasi
# =========================================================
elif page == "Evaluasi Model":
    st.subheader("Evaluasi dan pemilihan model")
    if not cf_comparison.empty:
        st.markdown("### Perbandingan Collaborative Filtering")
        st.dataframe(cf_comparison, use_container_width=True, hide_index=True)
        best_row = cf_comparison.sort_values("RMSE").iloc[0]
        st.info(
            f"Model dengan RMSE terendah pada tabel: **{best_row['Model']}** "
            f"(RMSE {best_row['RMSE']:.4f}; MAE {best_row['MAE']:.4f})."
        )

    if not alpha_evaluation.empty:
        st.markdown("### Evaluasi bobot alpha")
        alpha_col = "alpha"
        metric_options = [
            c for c in ["Precision@10", "Recall@10", "HitRate@10", "NDCG@10"]
            if c in alpha_evaluation.columns
        ]
        metric = st.selectbox("Pilih metrik ranking", metric_options, index=metric_options.index("NDCG@10") if "NDCG@10" in metric_options else 0)
        sorted_alpha = alpha_evaluation.sort_values(alpha_col)
        fig = px.line(
            sorted_alpha, x=alpha_col, y=metric, markers=True,
            title=f"{metric} pada berbagai nilai alpha",
            hover_data=[c for c in ["Precision@10", "Recall@10", "HitRate@10", "NDCG@10", "Evaluated Users"] if c in sorted_alpha.columns]
        )
        fig.update_layout(height=420, xaxis_title="Alpha (bobot CF)", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

        best_metric_row = alpha_evaluation.loc[alpha_evaluation[metric].idxmax()]
        st.metric(
            f"Alpha terbaik berdasarkan {metric}",
            f"{best_metric_row['alpha']:.1f}",
            help="Dipilih dari nilai tertinggi pada metrik yang sedang ditampilkan."
        )
        st.dataframe(sorted_alpha, use_container_width=True, hide_index=True)
        st.caption(
            "Metrik ranking ini mengikuti hasil evaluasi yang tersimpan. "
            "Nilai terbaik dapat berbeda antar-metrik."
        )

# =========================================================
# Halaman data
# =========================================================
elif page == "Data Buku & Rating":
    st.subheader("Eksplorasi data yang digunakan model")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Distribusi rating")
        rating_counts = model_ratings["Book-Rating"].value_counts().sort_index().rename_axis("Rating").reset_index(name="Jumlah")
        fig = px.bar(rating_counts, x="Rating", y="Jumlah", title="Jumlah rating per nilai")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Buku dengan rating terbanyak")
        top_books = (
            model_ratings.groupby("ISBN").agg(
                Jumlah_Rating=("Book-Rating", "size"),
                Rata_Rating=("Book-Rating", "mean")
            ).reset_index()
            .merge(candidate_books[["ISBN", "Book-Title"]].drop_duplicates("ISBN"), on="ISBN", how="left")
            .sort_values("Jumlah_Rating", ascending=False).head(15)
        )
        fig = px.bar(
            top_books.sort_values("Jumlah_Rating"),
            x="Jumlah_Rating", y="Book-Title", orientation="h",
            hover_data=["Rata_Rating"], title="15 buku dengan rating terbanyak"
        )
        fig.update_layout(height=520, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cari buku")
    search = st.text_input("Ketik judul, penulis, ISBN, atau penerbit")
    books_view = candidate_books.copy()
    searchable = [c for c in ["Book-Title", "Book-Author", "ISBN", "Publisher"] if c in books_view.columns]
    if search.strip() and searchable:
        mask = pd.Series(False, index=books_view.index)
        for col in searchable:
            mask |= books_view[col].fillna("").astype(str).str.contains(search.strip(), case=False, regex=False)
        books_view = books_view[mask]
    show_cols = [c for c in ["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"] if c in books_view.columns]
    st.dataframe(books_view[show_cols].head(500), use_container_width=True, hide_index=True)
    st.caption(f"Menampilkan maksimal 500 baris dari {len(books_view):,} hasil pencarian.")
    st.download_button(
        "⬇️ Unduh data buku yang ditampilkan",
        data=books_view[show_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="data_buku.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown(
    '<p class="small-note">Dashboard demonstrasi untuk model rekomendasi buku hybrid. '
    'Skor rekomendasi bersifat relatif untuk pengurutan kandidat, bukan probabilitas pembelian atau kepuasan.</p>',
    unsafe_allow_html=True
)
