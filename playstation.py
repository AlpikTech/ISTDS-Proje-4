#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# surprise kütüphanesini kontrol et
try:
    from surprise import Dataset, Reader, KNNBasic
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    # st.warning("Surprise kütüphanesi yüklü değil. İşbirlikçi filtreleme için basit korelasyon kullanılacak.")

# Veriyi yükle
try:
    df = pd.read_csv("video_game_reviews1.csv", delimiter=';')
except FileNotFoundError:
    st.error("Dosya bulunamadı. Lütfen 'video_game_reviews1.csv' dosyasının doğru konumunu kontrol edin.")
    st.stop()

# İçerik tabanlı analiz için metinleri birleştir
game_texts = df.groupby('Game Title')['User Review Text'].apply(lambda texts: " ".join(texts)).reset_index()
game_texts.rename(columns={'User Review Text': 'All Reviews Text'}, inplace=True)

# TF-IDF matrisi oluştur
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(game_texts['All Reviews Text'])

# Benzerlik matrisi
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(game_texts.index, index=game_texts['Game Title']).drop_duplicates()

# Kullanıcıları sayıya çevir
unique_users = {text: idx for idx, text in enumerate(df['User Review Text'].unique())}
df['UserID'] = df['User Review Text'].map(unique_users)
user_game_ratings = df.groupby(['UserID', 'Game Title'])['User Rating'].mean().reset_index()

# İşbirlikçi filtreleme matrisi
if SURPRISE_AVAILABLE:
    try:
        reader = Reader(rating_scale=(df['User Rating'].min(), df['User Rating'].max()))
        data = Dataset.load_from_df(user_game_ratings[['UserID', 'Game Title', 'User Rating']], reader)
        trainset = data.build_full_trainset()
        algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
        algo.fit(trainset)
        item_sim_matrix = algo.sim
        raw_to_inner = {trainset.to_raw_iid(i): i for i in range(trainset.n_items)}
    except Exception as e:
        st.warning(f"Surprise kütüphanesi hatası: {str(e)}. Basit korelasyon kullanılacak.")
        SURPRISE_AVAILABLE = False

if not SURPRISE_AVAILABLE:
    pivot = user_game_ratings.pivot(index='Game Title', columns='UserID', values='User Rating').fillna(0)
    game_corr = pivot.T.corr(min_periods=1).fillna(0)
    item_sim_matrix = game_corr.values

# Hibrit öneri fonksiyonu
def recommend_similar_games(game_title, top_n=5, alpha=0.5):
    idx = indices.get(game_title)
    if idx is None:
        return []

    content_scores = cosine_sim_matrix[idx].copy()

    if SURPRISE_AVAILABLE and game_title in raw_to_inner:
        inner_id = raw_to_inner[game_title]
        collab_scores = item_sim_matrix[inner_id]
    else:
        collab_scores = item_sim_matrix[idx] if idx < len(item_sim_matrix) else np.zeros(len(content_scores))

    # Normalizasyon
    content_scores_norm = content_scores / (np.max(content_scores) + 1e-10)
    collab_scores_norm = collab_scores / (np.max(collab_scores) + 1e-10)

    hybrid_score = alpha * collab_scores_norm + (1 - alpha) * content_scores_norm
    hybrid_score[idx] = -1  # Kendisini önermemek için

    top_indices = hybrid_score.argsort()[::-1][:top_n]
    return game_texts.iloc[top_indices]['Game Title'].tolist()

# Streamlit arayüzü
st.title("🎮 PlayStation Oyun Öneri Sistemi")
st.markdown("Kullanıcı yorumlarına ve puanlara göre oyun önerileri sunar.")
st.markdown("""# Ne kullandım:
- Cosinus Similarity
- Hybrid recommendation: `Content-Based Filtering` ve `Collaborative Filtering`
## `Collaborative Filtering` De Ne Kullandım:
- `Surprise` (Local only)
- `Basic Collaborative Filtering`
# Kendine Has Özellikler:
- Local Olarak Kolay Başlatma: `streamlit_start.bat`
---
""")

# Oyun seçimi
selected_game = st.selectbox("Bir oyun seçin:", sorted(df['Game Title'].unique()))

# Önerileri göster
if st.button("Oyun Önerilerini Göster"):
    if selected_game not in indices:
        st.error("Seçilen oyun veri setinde bulunamadı!")
    else:
        st.subheader("🎯 İçerik Tabanlı Öneriler")
        idx = indices[selected_game]
        sim_scores = cosine_sim_matrix[idx].copy()
        sim_scores[idx] = -1
        content_indices = sim_scores.argsort()[::-1][:5]
        for game in game_texts.iloc[content_indices]['Game Title']:
            st.write("🎯", game)

        st.subheader("🤝 Benzer Oyuncuların Tercihleri")
        if SURPRISE_AVAILABLE and selected_game in raw_to_inner:
            try:
                neighbors = algo.get_neighbors(raw_to_inner[selected_game], k=5)
                recs = [trainset.to_raw_iid(inner) for inner in neighbors]
            except Exception as e:
                st.warning(f"Surprise hatası: {str(e)}. Basit korelasyon kullanılıyor.")
                corr_series = pd.Series(item_sim_matrix[indices[selected_game]], index=game_texts['Game Title'])
                recs = corr_series.nlargest(6).index.tolist()[1:]  # Kendisini çıkar
        else:
            corr_series = pd.Series(item_sim_matrix[indices[selected_game]], index=game_texts['Game Title'])
            recs = corr_series.nlargest(6).index.tolist()[1:]  # Kendisini çıkar
        
        for game in recs[:5]:  # En fazla 5 öneri
            st.write("🤝", game)

        st.subheader("🧠 Karışık Öneri Sistemi (İçerik + Kullanıcı Bazlı)")
        hybrid_recs = recommend_similar_games(selected_game, top_n=5, alpha=0.5)
        for game in hybrid_recs:
            st.write("🧠", game)
st.markdown("""
---
# Fork Sayacı
[![Fork me on GitHub](https://img.shields.io/github/forks/AlpikTech/ISTDS-Proje-4?style=social)](https://github.com/AlpikTech/ISTDS-Proje-4)

""")

# In[ ]:




