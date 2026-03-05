import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Compatibility imports for LangChain v1.0+
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- UI Setup ---
st.set_page_config(page_title="SonicShield DocChat", page_icon="📄")
st.header("Smart Document 'Chatter' (RAG System)")

with st.sidebar:
    hf_token = st.text_input("Enter Hugging Face Token", type="password")
    st.info("Get a free token at huggingface.co/settings/tokens")

uploaded_file = st.file_uploader("Upload your UNIT-2 AI Notes", type="pdf")

if uploaded_file and hf_token:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
        with st.spinner("Processing document..."):
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.success("Notes Indexed!")

            # STABLE CHOICE: Llama-3.1-8B is highly compatible with ChatHuggingFace
            base_llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=hf_token,
                task="text-generation",
                temperature=0.5,
            )
            chat_llm = ChatHuggingFace(llm=base_llm)

            system_prompt = (
                "You are an AI assistant. Use the provided context to answer the question. "
                "Context: {context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            combine_docs_chain = create_stuff_documents_chain(chat_llm, prompt)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

        user_query = st.text_input("Ask a question about your AI Notes:")
        if user_query:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_query})
                st.markdown(f"**AI Response:** \n\n {response['answer']}")

    except Exception as e:
        st.error(f"Error: {e}")
elif not hf_token:
    st.warning("Please enter your Hugging Face token in the sidebar to begin.")