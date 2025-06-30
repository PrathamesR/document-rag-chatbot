import streamlit as st
from gemini.embeddings import store_file_info, do_embeddings_exist, get_top_k_chunks, do_tables_exist, query_with_gemini
from gemini.gemini_calls import call_gemini
import json

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.no_of_files = 0

# Title
st.title("Chatbot with Attachment")

# Load previous messages
for msg in st.session_state.chat_history:
    role = "assistant" if msg["role"] == "model" else msg["role"]
    with st.chat_message(role):
        st.markdown(msg["parts"][0]["text"])
        if "files" in msg:
            for i, file in enumerate(msg["files"]):
                st.download_button(
                    label=f"📎 {file.name}",
                    data=file.getvalue(),
                    file_name=file.name,
                    key=f"download_{i}_{file.name}"
                )
# Layout: Chat input and file uploader side by side
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.chat_input("Type your message here...")

upload_key = st.session_state.get("upload_key", 0)
with col2:
    uploaded_files = st.file_uploader(
                        "📎",
                        type=None,
                        label_visibility="collapsed",
                        accept_multiple_files=True,
                        key=f"file_uploader_{upload_key}"
                    )

# Handle user input
if user_input:
    print('------')
    user_msg = {"role": "user", "parts": [{"text": user_input}]}
    if uploaded_files:
        user_msg["files"] = uploaded_files
        for file in uploaded_files:
            st.session_state.no_of_files+=1
            store_file_info(file)
    st.session_state.chat_history.append(user_msg)

    retrieved_context = []
    if do_embeddings_exist():
        retrieved_context.append(get_top_k_chunks(user_input))
        print("top k for the query:", user_input)
        print(retrieved_context)

    print('=========')
    if do_tables_exist():
        print("query df with gemini")
        response = query_with_gemini(user_input)
        excel_data = "The following info has been fetched from user submitted documents - \n" + json.dumps(response)

        retrieved_context.append(excel_data)

    response = call_gemini(st.session_state.chat_history, retrieved_context)
    bot_msg = {"role": "model", "parts": [{"text": response}]}
    st.session_state.chat_history.append(bot_msg)

    st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
    st.rerun()
