import os
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
import json
import re
import streamlit as st

load_dotenv(find_dotenv())
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
else:
    print(api_key)
client = genai.Client(api_key=api_key)
gemini_chat_model = client.models.get(model="gemini-2.0-flash")

def get_gemini_image_info(image_bytes):
    # Dummy placeholder function
    return "Image content extracted by Gemini (simulated)"

def call_gemini(messages, retrieved_context=None, system_prompt=None):
    """
    Sends a chat message to the Gemini model with optional RAG context and system prompt.

    Args:
        messages (list): Chat history formatted with role and parts.
        retrieved_context (str, optional): Additional context (e.g., RAG output) to guide the response.
        system_prompt (str, optional): Custom system-level instructions for the model.

    Returns:
        str: The model's response text.
    """

    # print("USER MESSAGES", messages)
    # print("CONTEXT", retrieved_context)
    if system_prompt==None:
        system_prompt = """You are an intelligent assistant that answers user questions strictly based on the content extracted from uploaded documents.
If the relevant information is not present in the documents, politely respond that you don't have enough information to answer the question.
Avoid guessing or adding extra knowledge beyond the provided context. Be concise, factual, and professional in your tone.
Always assume the documents are accurate and up-to-date, and treat them as the sole source of truth. Provide the final response in a markdown format.
"""

    gemini_history = [
        {k: v for k, v in msg.items() if k in {"role", "parts"}}
        for msg in messages[:-1]
    ]

    if retrieved_context:
        gemini_history.append({
            "role": "user",
            "parts": [{
                "text": f"Use the following context to answer:\n\n{retrieved_context}"
            }]
        })

    chat_session = client.chats.create(
        model="gemini-2.0-flash",
        history=gemini_history,
        config=GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
    )

    # print("parts", messages[-1]['parts'][0]['text'])
    response = chat_session.send_message(messages[-1]['parts'][0]['text'])
    # print("API Res", response)
    # return f"You said: {messages[-1]['parts'][0]}"
    return response.text


def ask_gemini_what_to_query(user_query, tables):
    """
    Generates a Pandas query using Gemini to retrieve relevant data from user-submitted tables.

    Args:
        user_query (str): The user's natural language question.
        tables (list): List of dicts containing table metadata, summaries, and DataFrames.

    Returns:
        dict: A dictionary with 'df_name' and 'query' keys for executing the suggested Pandas expression.
    """
    table_data = "You have the following dataframes with you:\n\n"

    for i, table_obj in enumerate(tables):
        for table_name, table_data_dict in table_obj.items():
            metadata = table_data_dict["metadata"]
            summary = table_data_dict["table_summary"]
            df = table_data_dict["data_frame"]

            chunk = f"""
Dataframe {i+1}: "{table_name}"

"Schema":
{json.dumps(metadata, indent=2)}

"Summary":
{summary}

"Sample Rows":
{df.head(5).to_json(orient="records", indent=2)}
"""
            table_data += chunk + "\n"
    prompt = f"""
{table_data}

Respond with a Python Pandas expression that would extract the relevant data from the relevant dataframe.
Do not include imports or explanations, only the expression and df name in the format.
Always use `df` as the name of the dataframe in the query. 
Never use the file name or any other identifier.
{{
  "df_name": "...",
  "query": "df['Product'].unique()"  <-- important
}}


Now answer this user question:
"{user_query}"
"""
    response =  call_gemini([
        {"role": "user", "parts": [{"text": prompt}]}
    ], system_prompt="""
You are a smart and helpful assistant. Only return valid JSON containing the DataFrame name and a pandas query. Always reference columns using df['col']. 
{
  "df_name": "<string>",
  "query": "<pandas_query_string>"
}
Do not include markdown, explanation, or anything else.
""")

    print("Gemini df response", response)

    response_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())

    try:
        return json.loads(response_clean)
    except json.JSONDecodeError:
        st.error("Gemini returned invalid JSON:\n" + response)
        return {}
