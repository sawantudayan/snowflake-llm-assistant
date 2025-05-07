# app_streamlit.py
import streamlit as st

from semantic_search import search_semantic, get_all_metadata_filters

st.set_page_config(page_title="📚 Semantic Search RAG Assistant", layout="wide")
st.title("📚 Semantic Search over Document Chunks")

# Sidebar Filters
st.sidebar.header("🔍 Filters")
filenames, categories = get_all_metadata_filters()
selected_filenames = st.sidebar.multiselect("Filter by Filename", filenames)
selected_categories = st.sidebar.multiselect("Filter by Category", categories)
k = st.sidebar.slider("Top-K Results", min_value=1, max_value=20, value=5)

# Search Input
query = st.text_input("Enter your search query")

if query:
    filters = {}
    if selected_filenames:
        filters['filename'] = selected_filenames
    if selected_categories:
        filters['category'] = selected_categories

    with st.spinner("Running semantic search..."):
        results = search_semantic(query=query, k=k, filters=filters)

    if results:
        st.subheader(f"🔎 Top {len(results)} Matches")
        for result in results:
            highlighted = result['chunk_text'].replace(query, f"**{query}**")
            with st.expander(
                    f"📄 {result['filename']} | 📂 {result['category']} | 🧠 Similarity: {result['distance']:.2f}"):
                st.markdown(highlighted)
                st.caption(f"Chunk ID: {result['chunk_id']}")
                feedback_col1, feedback_col2 = st.columns([1, 1])
                with feedback_col1:
                    if st.button("👍 Relevant", key=f"up_{result['chunk_id']}"):
                        st.success("Thanks for the feedback!")
                        # Log positive feedback here
                with feedback_col2:
                    if st.button("👎 Irrelevant", key=f"down_{result['chunk_id']}"):
                        st.warning("Feedback noted.")
                        # Log negative feedback here
    else:
        st.warning("No relevant results found. Try refining your query or filters.")
