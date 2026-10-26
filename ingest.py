import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "data/Tata_SUVs_India_Feature_Guide_2026.pdf"
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Check the PDF filename inside the data folder."
        )

    print("1. Loading Tata PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} PDF pages.")

    print("2. Splitting PDF into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text was extracted. Check whether the PDF is scanned."
        )

    print(f"Created {len(chunks)} chunks.")

    print("3. Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("4. Creating local ChromaDB...")
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name="tata_car_guide"
    )

    print("\nSUCCESS!")
    print(f"PDF pages: {len(documents)}")
    print(f"Text chunks: {len(chunks)}")
    print(f"Vector database saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
