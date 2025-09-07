import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pdf2image import convert_from_bytes
import pytesseract

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache vectorstore so it doesn't reload every time
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def extract_text_from_pdf(file):
    try:
        st.info("📄 Extracting text from PDF...")
        reader = PdfReader(file)
        text = ""
        page_count = len(reader.pages)
        st.success(f"✅ PDF loaded successfully with {page_count} page(s).")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                st.write(f"Page {i+1}: Extracted {len(page_text)} characters.")
            else:
                st.warning(f"Page {i+1}: No text found on this page.")
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {e}")
        return None


def main():
    st.set_page_config(page_title="Medical Chatbot", layout="centered")
    st.title("🩺 Medical Chatbot")

    st.write("Ask me about medical symptoms or treatments! ⚠️ *Disclaimer: This is for educational purposes only.*")

    # Initialize vectorstore
    vectorstore = get_vectorstore()
    uploaded_file = st.file_uploader("📄 Upload a medical report (PDF)", type="pdf")

    if uploaded_file:
        st.info("Analyzing uploaded report...")
        report_text = extract_text_from_pdf(uploaded_file)
        if report_text:
            st.text_area("Extracted Report Text", report_text, height=300)

            # ---- Place the LLM summary code here ----




            prompt = PromptTemplate(
                input_variables=["report_text"],
                template="""
        You are a medical assistant.
        Here is a patient's report:

        {report_text}

        Summarize the abnormal findings and provide a simple, plain-language interpretation.
        """
            )

            chain = LLMChain(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.0,
                    groq_api_key=os.environ.get("GROQ_API_KEY"),
                ),
                prompt=prompt
            )

            summary = chain.run(report_text=report_text)
            st.subheader("🩺 Report Summary & Interpretation")
            st.write(summary)
        else:
            st.warning("⚠️ No text could be extracted from this PDF.")





    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )


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

        if "debug_logs" not in st.session_state:
            st.session_state.debug_logs = []





    # Show conversation history
    if st.session_state.memory.chat_memory.messages:
        st.divider()
        st.write("### 🗂 Conversation History")
        for message in st.session_state.memory.chat_memory.messages:
            role = "User" if message.type == "human" else "Bot"
            st.write(f"**{role}:** {message.content}")


if __name__ == "__main__":
    main()
