import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
import time

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.environ["GROQ_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]

# Set up Streamlit interface
st.title("Aurelius Suite - Conversational Chatbot")

# Initialize the language model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Define the conversational prompt template
prompt = ChatPromptTemplate.from_template(
"""
You are Aurelius Suite, an expert assistant enhancing the Aurelius application. Answer based solely on the provided knowledge base from Confluence, PDFs, and Knowledge Transition Videos. Search the entire knowledge base for each question, not just the current context or chat history. If no answer exists across all documents, say: "Unfortunately, I don’t have an answer for this. Kindly contact the Aurelius Team support."

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Instructions:
1. Use the full knowledge base to find accurate, relevant answers for every question.
2. Reference chat history only if the question explicitly builds on prior exchanges (e.g., "Tell me more about that").
3. Keep responses clear and concise, avoiding speculation beyond the documents.
4. If the answer isn’t found in any document, provide the fallback response above.
"""
)

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def vector_embedding(force_refresh=False):
    if "vectors" not in st.session_state or force_refresh:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        documents_path = "./documents"
        
        if not os.path.exists(documents_path):
            os.makedirs(documents_path)
            st.warning("Created documents directory. Please add your files.")
            return

        try:
            loader = DirectoryLoader(
                documents_path,
                glob="**/*.[tp][xd][tf]",
                show_progress=True,
                use_multithreading=True,
                silent_errors=False
            )
            loader.loader_cls_by_suffix = {
                ".txt": lambda path: TextLoader(path, encoding="utf-8"),
                ".pdf": PyPDFLoader
            }
            loader.loader_kwargs_by_suffix = {
                ".txt": {"encoding": "utf-8"},
                ".pdf": {}
            }
            docs = loader.load()
            if not docs:
                st.error("No documents found in the directory. Please add files to process.")
                st.session_state.vectors = None
                return
            st.session_state.docs = docs
            # Move document list to sidebar, not main UI
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            st.session_state.vectors = None
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)
        
        try:
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            st.session_state.vectors = None
            return

# Chat interface at the top
st.subheader("Chat with Aurelius Suite")

# Display chat history
if st.session_state.chat_history:
    for message in st.session_state.chat_history[:-1]:
        if message["role"] == "user":
            st.write(f"**You**: {message['content']}")
        else:
            st.write(f"**Aurelius Suite**: {message['content']}")

# User input
user_input = st.text_input("Ask your question here", key="user_input")

# Handle user input and show only the latest response
if user_input:
    if st.session_state.vectors is None:
        st.warning("Knowledge base not initialized. Please click 'Process Documents' first.")
    else:
        try:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10})
            retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
            chat_history_for_chain = [(msg["content"], next_msg["content"]) 
                                    for msg, next_msg in zip(st.session_state.chat_history[::2], 
                                                            st.session_state.chat_history[1::2]) 
                                    if msg["role"] == "user" and next_msg["role"] == "assistant"]
            start = time.process_time()
            response = retrieval_chain.invoke({
                "question": user_input,
                "chat_history": chat_history_for_chain
            })
            response_time = time.process_time() - start
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
            st.write(f"**You**: {user_input}")
            st.write(f"**Aurelius Suite**: {response['answer']}")
            st.write(f"Response time: {response_time:.2f} seconds")
            with st.expander("Source Documents"):
                for i, doc in enumerate(response["source_documents"]):
                    st.write(f"**Document {i + 1} (Source: {doc.metadata.get('source', 'Unknown')})**:")
                    st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

# Process documents buttons below chat
st.markdown("---")  # Separator for clarity
col1, col2 = st.columns(2)
with col1:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            vector_embedding(force_refresh=True)
        if st.session_state.vectors is not None:
            st.success("Documents processed successfully")
        else:
            st.error("Failed to initialize knowledge base. Check debug info in sidebar.")
with col2:
    if st.button("Refresh Knowledge Base"):
        with st.spinner("Refreshing knowledge base..."):
            vector_embedding(force_refresh=True)
        if st.session_state.vectors is not None:
            st.success("Knowledge base refreshed successfully")
        else:
            st.error("Failed to refresh knowledge base. Check debug info in sidebar.")

# Sidebar debug info
st.sidebar.markdown("### Debug Info")
if "docs" in st.session_state:
    st.sidebar.write(f"Loaded documents: {len(st.session_state.docs)}")
    with st.sidebar.expander("Document List"):
        for doc in st.session_state.docs:
            st.sidebar.write(f"- {doc.metadata.get('source', 'Unknown')}")
else:
    st.sidebar.write("No documents loaded yet.")
if "final_documents" in st.session_state:
    st.sidebar.write(f"Total chunks: {len(st.session_state.final_documents)}")
st.sidebar.markdown("""
### Instructions
1. Click 'Process Documents' to load the knowledge base
2. Ask questions in the chat box
3. Use 'Refresh Knowledge Base' to update if needed
""")