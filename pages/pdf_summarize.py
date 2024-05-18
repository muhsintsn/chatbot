import os
import streamlit as st
import tempfile
import google.generativeai as ggi
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import  GoogleGenerativeAI, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
import textwrap 
import warnings;
warnings.filterwarnings('ignore')


#st.title("Chatbot Application using OpenAI gpt-3.5-turbo")

#main header
html_temp = """
<div style="background-color:orange;padding:10px">
<h3 style="color:white;text-align:center;">Pdf File Summarize</h3>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.write("----")

#api key
with st.sidebar:
    google_api_key = st.text_input("Google API Key", key="file_qa_api_key", type="password")

if google_api_key:
    ggi.configure(api_key=google_api_key)
    st.success("Google API key has been successfully set!")
else:
    st.error("Please enter a valid Google API key.")


def load_read_pdf():
    global pdf
    # streamlit file uploader
    uploaded_file=st.file_uploader('Choose a pdf file', type='pdf')
    if uploaded_file is not None:
        try:
            # save the file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
        
            # read pdf using PyPDFium2Loader
            pdf = get_pdf(tmp_file_path)
            if pdf:
                st.success('Success!')
                

        except Exception as e:
            st.error(f"An error occured: {e}")


def get_pdf(file_path):
    try:
        if not os.path.exists(file_path):
            raise ValueError(f"The file does not exist in this path: {file_path} ")

        # make sure that the file is pdf
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("The file must be a .pdf file")

        # loading the pdf file using PyPDFium2Loader
        file_loader = PyPDFium2Loader(file_path)
        pdf_documents = file_loader.load()

        return pdf_documents
    
    except Exception as e:
        st.error(f"Error in reading the file: {e}")
        return None

def stuff_model(pdf):
    llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-1.5-pro-latest',max_tokens=1024, google_api_key=google_api_key)
    chain = load_summarize_chain(
    llm,
    chain_type='stuff',
    #prompt=prompt,
    verbose=False
)  
    chain.invoke(pdf[0:5])['output_text']

def map_reduce_model(pdf):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    chunks = text_splitter.split_documents(pdf)



st.title(":orange[Pdf Summarizing using Gemini 1.5 Pro]")
load_read_pdf()
st.write("----")

btn = st.button("Summary")
if btn:
    result =stuff_model(pdf)
    st.subheader("Summary : ")
    st.write(result)