import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="🍜 District 5 Food Recommender", page_icon="🍲", layout="wide")

st.title("🍜 District 5 Food Recommender")
st.caption("Find your perfect dish in Chợ Lớn, Hồ Chí Minh City")


@st.cache_data
def load_data():
    df = pd.read_csv("district5_dishes_dishlist.csv")
    return df

df = load_data()


@st.cache_resource
def load_model():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model

model = load_model()


@st.cache_resource
def load_embeddings():
    embeddings = np.load("dish_embeddings.npy")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

index, embeddings = load_embeddings()


query = st.text_input("🍽 What are you craving?", placeholder="e.g. spicy noodle soup, sweet dessert, crispy snack...")
top_k = 5


if query:
    # Encode and normalize 
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # Search in FAISS index
    D, I = index.search(q_emb, top_k)

    st.subheader("🥢 Recommended Dishes:")
    for rank, idx in enumerate(I[0]):
        dish = df.iloc[idx]
        st.markdown(f"### {rank+1}. {dish['name_vn']} / {dish['name_en']}")
        st.markdown(f"**Taste Profile:** {dish['taste_profile']}")
        st.markdown(f"**Description:** {dish['description_en']}")
        st.markdown(f"**Main Ingredients:** {dish['min_ingredients']}")
        score = D[0][rank] * 100
        color = "green" 
        st.markdown(
                f"<b>This dish matches your description by "
                f"<span style='color:{color};'>{score:.1f}%</span></b>",
                unsafe_allow_html=True
        )
        st.divider()

else:
    st.info("👆 Type in a craving above to get personalized recommendations.")


