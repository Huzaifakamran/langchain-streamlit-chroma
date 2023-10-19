from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def store_embeddings(chunks):
    embedding_function = OpenAIEmbeddings()
    embeddings = Chroma.from_texts(chunks, embedding_function, persist_directory="./chroma_db")
    return embeddings

def get_embeddings():
    embedding_function = OpenAIEmbeddings()
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    return db3

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask your remote database 💬")
    db3 = get_embeddings()
    # create chain 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=db3.as_retriever()
    )

    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your pdfs here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                embeddings = store_embeddings(chunks)

if __name__ == '__main__':
    main()