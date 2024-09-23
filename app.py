
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

# Load the sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display the name of the uploaded file
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Read the PDF file
    pdf_reader = PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    st.write(f"The PDF has {num_pages} pages.")

    # Extract text from each page
    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text
        else:
            st.warning(f"Page {page_num + 1} has no extractable text.")

    if text:
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        st.write(f"Text split into {len(chunks)} chunks.")

        # Generate sentence embeddings for each chunk
        chunk_embeddings = model.encode(chunks)

        st.write("Embeddings generated successfully!")

        # Let the user ask a question
        query = st.text_input("Ask a question about the PDF:")

        if query:
            # Embed the user query
            query_embedding = model.encode([query])

            # Compute cosine similarities between query embedding and chunk embeddings
            similarities = cosine_similarity(query_embedding, chunk_embeddings)

            # Find the index of the most similar chunk
            most_similar_idx = np.argmax(similarities)

            # Retrieve the most relevant chunk
            relevant_chunk = chunks[most_similar_idx]

            st.write("Response:")
            st.write(relevant_chunk)

    else:
        st.error("No text extracted from the PDF.")
else:
    st.write("No file uploaded yet. Please upload a PDF file.")

if __name__ == "__main__":
    main()
