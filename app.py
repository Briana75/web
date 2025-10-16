import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@st.cache_data
def load_data():
    df = pd.read_excel("District5_Food_List.xlsx")
    return df

df = load_data()


@st.cache_resource
def load_model():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  
    return model

model = load_model()


@st.cache_resource
def build_faiss_index(descriptions):
    embeddings = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_faiss_index(df["Description"].tolist())

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🍜 District 5 Food Recommendation")
st.write("Type what kind of food you're craving")
st.write("When you get what you want, you can search it online for more info")
query = st.text_input("Describe your craving (e.g., 'sweet dessert', 'spicy soup', 'crispy snack')")

if query:
    # Encode query
    query_emb = model.encode([query], convert_to_numpy=True)
    # Search FAISS
    k = 5
    distances, indices = index.search(query_emb, k)

    st.subheader("🍽 Recommended Foods:")
    for rank, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        st.markdown(f"### {rank+1}. {row['Tên món ăn']}")
        st.markdown(f"**Description:** {row['Description']}")
        st.markdown(f"**Taste:** {row['Taste']}")
        st.markdown(f"**Main Ingredients:** {row['Main Ingredients']}")
        st.markdown(f"**Where to try:** {row['Suggested Location']}")
        st.markdown(f"_(Similarity: {1/(1+distances[0][rank]):.2f})_")
        st.markdown("---")
        st.write(distances)
else:
    st.info("👆 Type something like 'sweet dessert' or 'noodle soup' above to get recommendations.")
