import os
import streamlit as st
import google.generativeai as ggi
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import  GoogleGenerativeAI, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.chains.question_answering import load_qa_chain
import textwrap


#main header
html_temp = """
<div style="background-color:orange;padding:10px">
<h3 style="color:white;text-align:center;">Pdf File Q&A Chatbot</h3>
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
                init_model(pdf)

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

def init_model(pdf):
    global index, loaded_index, chain
    # document splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    text_splitter.split_documents(pdf)

    # word embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type='retrieval_document', google_api_key=google_api_key)

    # vectorstores
    index = Chroma().from_documents(documents=pdf,
                                embedding=embeddings,
                                persist_directory="/Users/muhsi/Desktop/Streamlit_chatbot/vectorstore")

    loaded_index = Chroma(persist_directory="/Users/muhsi/Desktop/Streamlit_chatbot/vectorstore",
                        embedding_function=embeddings)

    llm = GoogleGenerativeAI(
    model="gemini-1.5-pro-latest",temperature=0,google_api_key=google_api_key,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)
    chain=load_qa_chain(llm, chain_type="stuff")


def retrieve_query(query,k=5):
    matching_results=index.similarity_search(query,k=k)
    return matching_results

def get_answers(query):
    doc_search=retrieve_query(query)
    response=chain.invoke(input={"input_documents":doc_search, "question":query})["output_text"]
    wrapped_text = textwrap.fill(response, width=100)
    return wrapped_text

st.title(":orange[Pdf Question and Answer using Gemini 1.5 Pro]")

load_read_pdf()
st.write("----")

user_quest = st.text_input("Ask a question:")
btn = st.button("Ask")

if btn and user_quest:
    result = get_answers(user_quest)
    st.subheader("Response : ")
    st.write(result)













