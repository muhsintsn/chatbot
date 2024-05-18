import os
import streamlit as st
import tempfile
import google.generativeai as ggi
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader
from langchain_google_genai import  GoogleGenerativeAI, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
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

def stuff_model(pdf, google_api_key):
    
    llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-1.5-pro-latest',max_tokens=1024, google_api_key=google_api_key)
    chain = load_summarize_chain(
    llm,
    chain_type='stuff',
    #prompt=prompt,
    verbose=False
)  
    output_summary=chain.invoke(pdf[0:5])['output_text']
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)


def map_reduce_model(pdf,google_api_key):
    llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-1.5-pro-latest',max_tokens=1024, google_api_key=google_api_key)
  
    final_combine_prompt='''
     Provide a final summary of the entire text with at least 1000 words.
     Add a Generic  Title,
     Start the precise summary with an introduction and provide the
     summary in bullet points for the text.
     text: '{text}'
     summary:
            '''
    final_combine_prompt_template=PromptTemplate(input_variables=['text'],
                                             template=final_combine_prompt)
    chain = load_summarize_chain(llm,
                             chain_type="map_reduce", combine_prompt=final_combine_prompt_template)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

    chunks = text_splitter.split_documents(pdf)
    
    output_summary=chain.invoke(chunks)["output_text"]
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)

def refine_model(pdf, google_api_key):
    llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-1.5-pro-latest',max_tokens=1024, google_api_key=google_api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    chunks = text_splitter.split_documents(pdf)
    chain = load_summarize_chain(llm,
                             chain_type="refine")
    output_summary=chain.invoke(chunks)["output_text"]
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)    



st.title(":orange[Pdf Summarizing using Gemini 1.5 Pro]")
load_read_pdf()
st.write("----")

options = st.radio(
    "Choose a summary type:",
    ["Summary of Certain Pages", "Short Summary of the Entire Document", "Detailed Summary of the Entire Document"]
)

btn = st.button("Get Summary")
if btn:
    if options=="Summary of Certain Pages":
       result = stuff_model(pdf, google_api_key)
       st.text(result)
    elif options=='Short Summary of the Entire Document':
        result = map_reduce_model(pdf,google_api_key) 
        st.text(result)
    elif options== "Detailed Summary of the Entire Document":
        result= refine_model(pdf,google_api_key)   
        st.text(result)

    #st.subheader("Summary : ")
    #st.write(result)