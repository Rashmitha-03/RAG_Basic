import streamlit as st
from rag_core import ingest_pdf, retrieve, answer_stream, collection

st.set_page_config(page_title="DocuChat", page_icon="📄")
st.title("📄 DocuChat: chat with your PDFs")


def show_sources(hits):
    with st.expander("Sources"):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}]** {h['source']}, page {h['page']} (similarity {h['score']:.2f})")
            st.caption(h["text"])


with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Index documents"):
        with st.spinner("Chunking and embedding..."):
            for f in files:
                n = ingest_pdf(f.getvalue(), f.name)
                if n:
                    st.success(f"{f.name}: {n} chunks")
                else:
                    st.warning(f"{f.name}: no text found (scanned PDF?)")
    st.caption(f"Chunks in database: {collection.count()}")
    k = st.slider("Chunks to retrieve (k)", 1, 8, 4)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if collection.count() == 0:
        st.warning("Upload and index a PDF first.")
    else:
        hits = retrieve(question, k=k)
        with st.chat_message("assistant"):
            answer = st.write_stream(answer_stream(question, hits))
            show_sources(hits)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": hits}
        )