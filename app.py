import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com)')

def main():
    st.header("Chatbot")

# Main content
st.title("Upload a PDF File")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display the name of the uploaded file
    st.write(uploaded_file.name)

    # Read the PDF file
    pdf_reader = PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    st.write(f"The PDF has {num_pages} pages.")

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Load the sentence embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another model from the sentence-transformers library

    # Generate sentence embeddings for each chunk
    chunk_embeddings = model.encode(chunks)

    st.write("Generated Embeddings:")
    st.write(chunk_embeddings)

else:
    st.write("No file uploaded yet. Please upload a PDF file.")

if __name__ == "__main__":
    main()
