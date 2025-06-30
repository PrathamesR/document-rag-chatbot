import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
import fitz
import streamlit as st
from google import genai
from google.genai  import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

from .gemini_calls import get_gemini_image_info, call_gemini, ask_gemini_what_to_query

load_dotenv(find_dotenv())
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
else:
    print(api_key)
client = genai.Client(api_key=api_key)


def store_file_info(file):
    """
    Processes an uploaded file and stores relevant information.

    For text files, generates and stores chunk embeddings using Gemini.
    For structured Excel files, extracts metadata, summarizes each sheet, and stores the info.

    Parameters:
        file (UploadedFile): The uploaded file from Streamlit.
    """
    data = extract_text_from_uploaded_file(file)
    # print(data)
    if isinstance(data, str): # or detect_excel_type(data)=="textual":
        chunks = chunk_text(data)
        for chunk in chunks:
            embedding = client.models.embed_content(
                model="text-embedding-004",
                # model="gemini-embedding-exp-03-07",
                contents=chunk,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            chunk_embeddings.append({
                "text": chunk,
                "embedding": embedding.embeddings[0].values  
            })

        print("Embeddings Generated:", len(chunk_embeddings))
        st.success(f"Generated {len(chunk_embeddings)} embeddings for {file.name}")
    
    elif isinstance(data, dict):
        for sheet_name, df in data.items():
            metadata = extract_table_metadata(df)
            summary = summarize_table_with_gemini(file.name+'-'+sheet_name, df)
            table_data.append({file.name:{"metadata":metadata, "table_summary":summary, "data_frame": df}})
            # print(table_data)
            st.success(f"Stored table metadata for {file.name}")

def query_with_gemini(user_query):
    """
    Executes a Gemini-generated query on the matching dataframe.

    Retrieves the appropriate table, runs the query using `eval`, and returns the result with context.

    Parameters:
        user_query (str): The user’s question to be translated into a dataframe query.
    
    Returns:
        str: A formatted string with the data source, executed query, table summary, and result.
    
    Raises:
        ValueError: If Gemini response lacks required keys.
        RuntimeError: If query execution fails.
        KeyError: If the specified dataframe is not found.
    """
    query_info = ask_gemini_what_to_query(user_query, table_data)
    df_name, query = query_info.get("df_name", "").lower(), query_info.get("query")

    if not df_name or not query:
        raise ValueError("Gemini must return 'df_name' and 'query'.")

    for table in table_data:
        for name, info in table.items():
            if df_name in {name.lower(), os.path.splitext(name.lower())[0]}:
                df = info["data_frame"].copy()
                df = df.astype(str) if "object" in df.dtypes.values else df

                try:
                    result = eval(query, {}, {"df": df, "pd": pd})
                    return (
                        f"Data Source: {name}\n"
                        f"Query Executed: `{query}`\n"
                        f"Table Summary: {info['table_summary']}\n\n"
                        f"Query Result:\n{result}"
                    )
                except Exception as e:
                    raise RuntimeError(f"❌ Error executing query:\n`{query}`\n\n{e}")

    raise KeyError(f"DataFrame matching '{df_name}' not found.")


def chunk_text(text, chunk_size=2000, chunk_overlap=200):
    """Splits text into overlapping chunks using semantic-aware separators."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def cosine_similarity(a, b):
    """Computes cosine similarity between two vectors."""

    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_text_from_uploaded_file(uploaded_file):
    """Extracts content from uploaded text, PDF, or Excel file."""

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    elif filename.endswith(".pdf"):
        return extract_text_from_pdf_with_image_handling(uploaded_file)

    if filename.endswith(".csv"):
        st.error("Sorry we do not support .csv files as of now")
        df = pd.read_csv(uploaded_file)
        return df

    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file, sheet_name=None)
        return df

    else:
        raise ValueError("Unsupported file type")
    
def extract_text_from_pdf_with_image_handling(uploaded_file):
    """Extracts text and image info from each page of a PDF file."""

    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_content = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_content.append(f"📄 Page {page_num + 1} Text:\n{text}")

        # Process images on the page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Simulate image processing
            image_info = get_gemini_image_info(image_bytes)
            full_content.append(f"🖼️ Page {page_num + 1} Image {img_index + 1} Info:\n{image_info}")

    return "\n\n".join(full_content)

def detect_excel_type(df):
    """Classifies a DataFrame as 'textual' or 'structured' based on its shape."""

    if df.shape[0] < 5 and df.shape[1] < 3:
        return "textual"
    else:
        return "structured"


chunk_embeddings = []
table_data = []

def do_embeddings_exist():
    return len(chunk_embeddings)>0

def do_tables_exist():
    return len(table_data)>0

def get_top_k_chunks(user_query, k=2):
    """Returns the top-k most relevant text chunks based on cosine similarity with the user query."""

    scored = []

    query_embedding = client.models.embed_content(
        model="text-embedding-004",
        contents=user_query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    ).embeddings[0].values  

    for entry in chunk_embeddings:
        score = cosine_similarity(query_embedding, entry["embedding"])
        scored.append((score, entry["text"]))
    
    # Sort by similarity (highest first)
    scored.sort(reverse=True, key=lambda x: x[0])
    
    filtered = [entry for entry in scored if entry[0] >= 0.5]

    # print(scored[:k])
    return filtered[:k]


def extract_table_metadata(df: pd.DataFrame):
    """Extracts column names and their data types from a DataFrame."""

    return {col: str(df[col].dtype) for col in df.columns}

def summarize_table_with_gemini(table_name: str, df: pd.DataFrame):
    """Creates a summary for provided dataframe using gemini"""
    import json

    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Convert to plain Python types to ensure JSON-safe
    sample_rows = df.head(5).astype(str).to_dict(orient="records")

    prompt = f"""
You are given a table called "{table_name}".

Schema:
{json.dumps(schema, indent=2)}

Sample rows:
{json.dumps(sample_rows, indent=2)}

Based on the schema and data, summarize briefly what this table seems to represent. Mention column types and give an idea of its purpose or structure. If possible, suggest what questions this table could help answer.
"""
    
    TABLE_SUMMARY_SYSTEM_PROMPT = TABLE_SUMMARY_SYSTEM_PROMPT = """
You are a helpful data analyst.

Given a table's column schema and sample rows, your job is to describe:
1. What the table likely represents (its purpose or domain),
2. A brief overview of what kind of data is in each column,
3. Examples of questions this data could help answer.

Avoid assumptions not backed by the schema or rows. Keep it concise, structured, and insightful.
"""


    return call_gemini(
        [{"role": "user", "parts": [{"text": prompt}]}],
        system_prompt=TABLE_SUMMARY_SYSTEM_PROMPT
    )