
import os
from langchain_community.document_loaders import TextLoader,PyPDFLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_docs(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    #Check if the path exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    #Load the pdf from the docs directory
    loader = DirectoryLoader (
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError (f"No .pdf files exists in {docs_path}. Upload documents")


    
    # for i, doc in enumerate(documents):
    #     print(f"{i}:{doc.metadata['source']}")
    
    return documents


def split_documents(documents,chunk_size=800,chunk_overlap=0):
    """ Split documents into smaller size"""
    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks= text_splitter.split_documents(documents)

    # if chunks:
    #     for i, chunk in enumerate(chunks [:5]):
    #         print(f"\n --Chunk {i+1}---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length:{len(chunk.page_content)}characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-"*50)

    #     if len(chunks) >5:
    #         print(f"\n ...and {len(chunks)-5} more chunks")
    return chunks


def create_vector_store(chunks,persist_directory="db/chroma_db"):
    "Create the ChromaDB vector store"
    print("Creating embeddings and storing in ChromaDB")
    
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")

    #Create ChromaDB vectore store
    print("----Creating vector store----")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("---Finished creating vector store--")
    print(f"Vector store created and saved {persist_directory}")
    return vectorstore
    





def main():
    #1. Load files
    document=load_docs(docs_path="docs")

    #2.Chunk files
    chunks=split_documents(document)

    #3.Embedding and Storing in Database
    vectorstore=create_vector_store(chunks)
    


if __name__=="__main__":
    main()


