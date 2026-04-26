# 🤖 RAG Document Chatbot

An AI-powered chatbot that answers questions from your own PDF documents using **Retrieval Augmented Generation (RAG)**.

Built with Python · LangChain · ChromaDB · OpenAI · Streamlit

---

## ✨ Features

- 📄 Upload any PDF document
- 💬 Ask questions in natural language
- 🔍 Answers grounded in your document — no hallucinations
- 📎 Shows source passages used to generate each answer
- 🧠 Multi-turn conversation memory
- ⚙️ Adjustable chunk size and overlap settings
- 🚀 Clean, responsive Streamlit UI

---

## 🏗️ How It Works

```
User uploads PDF
      ↓
Document split into chunks (LangChain TextSplitter)
      ↓
Chunks embedded → stored in ChromaDB (Vector DB)
      ↓
User asks question
      ↓
Relevant chunks retrieved from ChromaDB
      ↓
Chunks + question sent to GPT-3.5
      ↓
Accurate answer returned with sources
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | OpenAI text-embedding-ada-002 |
| PDF Parsing | PyPDF |
| UI | Streamlit |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use the app
- Enter your OpenAI API key in the sidebar
- Upload a PDF file
- Click **Process Document**
- Start asking questions!

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 💡 Example Use Cases

- Chat with research papers
- Query company policy documents
- Study from textbook chapters
- Analyse business reports

---

## 🔑 API Key

You need an OpenAI API key. Get one at [platform.openai.com](https://platform.openai.com).

---

## 👩‍💻 Author

**Jayshree Rajpurohit**
[LinkedIn](https://linkedin.com/in/jayshreerajpurohit) · [GitHub](https://github.com/jayshreerajpurohit)
