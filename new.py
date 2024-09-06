'''import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from io import StringIO
import PyPDF2

# Function to generate summary using sumy
def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join([str(sentence) for sentence in summary])

# Function to process the uploaded file
def process_file(uploaded_file):
    # Check if the uploaded file is a PDF or text file
    if uploaded_file.type == 'application/pdf':
        pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
        text = ""
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extract_text()
    elif uploaded_file.type == 'text/plain':
        text = uploaded_file.read().decode("utf-8")
    else:
        text = "Unsupported file type."
    return text

st.title("Smart Student Hub")
st.write("Upload your lecture notes or PDFs and get summaries, flashcards, and quizzes!")

uploaded_file = st.file_uploader("Upload your notes (Text or PDF)", type=['txt', 'pdf'])

if uploaded_file is not None:
    text = process_file(uploaded_file)
    
    st.write("Uploaded Notes:")
    st.write(text[:1000])  # Display the first 1000 characters of the uploaded text
    
    if st.button("Generate Summary"):
        if text:
            summary = generate_summary(text)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.write("No text to summarize.")
    
    # Placeholder functions for flashcards and MCQs
    if st.button("Generate Flashcards"):
        flashcards = [("Sample Question", "Sample Answer")]
        st.subheader("Flashcards:")
        for question, answer in flashcards:
            st.write(f"Q: {question}")
            st.write(f"A: {answer}")
    
    if st.button("Generate MCQs"):
        mcqs = [{"question": "Sample Question?", "choices": ["Option A", "Option B"], "correct": "Option A"}]
        st.subheader("MCQs:")
        for mcq in mcqs:
            st.write(mcq['question'])
            for choice in mcq['choices']:
                st.write(f"- {choice}")



import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from .nltk.summarization.lexrank import summarize  

st.title("Text Summarizer")

uploaded_file = st.file_uploader("Upload your text file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read()

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Summarize the text using LexRank
    summary = summarize(sentences, ratio=0.2)  # Adjust the ratio as needed

    st.write("**Summary:**")
    st.write(summary)'''
import os

print(os.environ.get("OPENAI_API_KEY"))

import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

import os  # For accessing environment variables


def generate_response(txt):
    # Access API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    # ... rest of the function logic for splitting, creating documents, and summarization

# Page title
st.set_page_config(page_title=' Text Summarization App')
st.title(' Text Summarization App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form
result = []
with st.form('summarize_form', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not txt_input)
    submitted = st.form_submit_button('Summarize')
    if submitted and openai_api_key.startswith('sk-'):
        try:
            with st.spinner('Calculating...'):
                response = generate_response(txt_input)
                result.append(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            result.clear()  # Clear any partial results

if len(result):
    st.info(result[0])  # Assuming you only want to display the first summary