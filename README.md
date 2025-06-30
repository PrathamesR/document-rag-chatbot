# 📎 Gemini-Powered Document Chatbot with Structured Querying

An intelligent Streamlit-based chatbot that supports file uploads (PDFs, Excel), generates semantic embeddings using Google's Gemini API, performs RAG (Retrieval-Augmented Generation), and even executes Pandas queries over structured data using LLM-driven logic.

---

## ✨ Features

- 📄 Upload PDFs and Excel files (.xls, .xlsx)
- 🧠 Generate semantic embeddings using `text-embedding-004`
- 📊 Summarize Excel tables using Gemini
- 🤖 LLM-based structured querying using Pandas over uploaded tabular data
- 🧩 RAG-based context injection for conversational relevance
- 📎 Attachment preview and contextual memory per session
- 🔐 API Key securely loaded from `.env` file
- 🤖 Uses `gemini-2.0-flash` to generate responses quickly
---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Create a `.env` file in the root directory and add the following line:

```
GOOGLE_API_KEY=your-api-key-here
```

Create your own gemini api key here: https://aistudio.google.com/app/apikey

### 3. Run the app

```bash
streamlit run app.py
```