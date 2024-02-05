import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
# Configuration of google gemini pro
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#this function is regarding reading the pdf and extracting each page from that
def get_pdf_text(pdf_docs):
    text=""
    # reading all the pdf pages
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        #reading multiple pages
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

#this function divide all the text into chunks
def get_text_chunks(text):
    #divding text into chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#vector embedding it
def get_vectore_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectore_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vectore_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context , make sure that the a
    answer is appropriate, if answer is not available , just say "Sorry i dont thave the answer"
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

#generating response based on user input
def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index",embeddings)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    response=chain(
        {"input_documents":docs,"question":user_question}
        ,return_only_outputs=True
    )
    print(response)
    st.write("Reply:",response["output_text"])

def main():
        st.set_page_config("Chat with Multiple PDF")
         # Upload PDF files
        pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if pdf_files:
            # Process PDF files and get text chunks
            text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(text)

            # Generate and save vector store
            get_vectore_store(text_chunks)

            # User input section
            user_question = st.text_input("Ask a question:")
            if st.button("Generate Response"):
                if user_question:
                    user_input(user_question)
                else:
                    st.warning("Please enter a question.")

if __name__ == "__main__":
    main()