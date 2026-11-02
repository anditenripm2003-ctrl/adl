BOOK RECOMMENDATION STREAMLIT DASHBOARD

Isi paket:
- app.py: aplikasi dashboard Streamlit
- file artefak model: cf_model.joblib, tfidf_vectorizer.joblib, tfidf_matrix.npz,
  model_ratings.pkl, candidate_books.pkl, cf_comparison.csv,
  alpha_evaluation.csv, metadata.json
- requirements.txt: dependensi Python

CARA MENJALANKAN (Windows)
1. Ekstrak seluruh isi ZIP ke satu folder.
2. Buka Terminal/Command Prompt di folder tersebut.
3. Buat virtual environment (opsional tetapi disarankan):
   python -m venv .venv
   .venv\Scripts\activate
4. Instal dependensi:
   pip install -r requirements.txt
5. Jalankan:
   streamlit run app.py
6. Browser akan membuka dashboard Streamlit. Jika tidak terbuka otomatis,
   buka alamat lokal yang ditampilkan terminal (biasanya http://localhost:8501).

CATATAN
- Semua file artefak harus berada satu folder dengan app.py.
- Model Collaborative Filtering dibuat dengan Surprise; instalasi scikit-surprise
  dapat memerlukan compiler/build tools pada sebagian lingkungan Windows.
- Aplikasi ini menggunakan artefak hasil training dan tidak melatih ulang model.
