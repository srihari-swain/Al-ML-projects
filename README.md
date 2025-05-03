# Al-ML-projects
A collection of AI/ML projects and notebooks exploring algorithms, model building, and real-world use cases.
# Web Content Q&A Tool

A Python-based application that scrapes webpage content, creates embeddings, and answers questions based on the ingested information using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Features

- **Web Scraping** – Scrape main content from one or more webpage URLs.
- **Embeddings + Vector Store** – Convert content to embeddings using HuggingFace models and store in FAISS.
- **RAG-based Q&A** – Uses LangChain with Groq's Llama 3 model to answer questions based only on ingested content.

---

## 🧠 Technical Stack

| Component           | Technology Used                 |
|---------------------|----------------------------------|
| Web Scraping        | BeautifulSoup4 + Requests        |
| Embedding Model     | all-MiniLM-L6-v2 (via HF)        |
| Vector Database     | FAISS                            |
| RAG Framework       | LangChain + LangChain Community  |
| LLM API             | Groq (Llama 3) via `langchain-groq` |

---

## 📦 Installation & Running the App

Follow the steps below to install and run the application:

### 1. Clone the Repository
```bash
git clone https://github.com/srihari-swain/Al-ML-projects.git

```
### 2. Go to main dir
```
cd web-content-qa

```
### 3. Set your Groq API key as an environment variable in retriver.py 

### 4.Run the main script:
```
python main.py
```

