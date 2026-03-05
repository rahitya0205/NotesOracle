# 🦉 NotesOracle: Intelligent RAG Assistant

A Retrieval-Augmented Generation (RAG) system built to transform static technical documents into interactive AI agents.

## 🚀 Features
- **Semantic Retrieval:** Uses FAISS to index and retrieve precise context from PDFs.
- **Zero Hallucination:** Grounded in local data; verified using AI problem-solving agent characteristics.
- **Technical Logic:** Capable of retrieving complex formulas like the AO* evaluation function: $f(n) = g(n) + h(n)$.

## 🛠️ Tech Stack
- **Orchestration:** LangChain / LangChain-Classic
- **LLM:** Meta Llama-3.1-8B-Instruct (Hugging Face)
- **Vector Store:** FAISS
- **Frontend:** Streamlit

## ⚙️ How to Run
1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Enter your Hugging Face Token in the sidebar and upload a PDF.