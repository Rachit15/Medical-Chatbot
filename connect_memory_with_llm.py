import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache vectorstore so it doesn't reload every time
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def main():
    st.set_page_config(page_title="Medical Chatbot", layout="centered")
    st.title("🩺 Medical Chatbot")

    st.write("Ask me about medical symptoms or treatments! ⚠️ *Disclaimer: This is for educational purposes only.*")

    # Initialize vectorstore
    vectorstore = get_vectorstore()

    # ✅ Create memory to store conversation
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    # ❌ DO NOT pass memory directly to the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.0,
            groq_api_key=os.environ.get("GROQ_API_KEY"),
        ),
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True  # <-- returns both answer + docs
    )

    # Session state to store messages for UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat input box
    user_prompt = st.chat_input("Describe your symptoms or ask a medical question...")
    if user_prompt:
        # Show user message
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Get response from chain
        response = qa_chain.invoke({
            "question": user_prompt,
            "chat_history": st.session_state.memory.chat_memory.messages
        })
        bot_answer = response["answer"]

        # ✅ Manually save ONLY the answer to memory
        st.session_state.memory.save_context(
            {"question": user_prompt},
            {"answer": bot_answer}
        )

        # Show bot message
        st.chat_message("assistant").markdown(bot_answer)
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})

    # Show conversation history
    if st.session_state.memory.chat_memory.messages:
        st.divider()
        st.write("### 🗂 Conversation History")
        for message in st.session_state.memory.chat_memory.messages:
            role = "User" if message.type == "human" else "Bot"
            st.write(f"**{role}:** {message.content}")


if __name__ == "__main__":
    main()
