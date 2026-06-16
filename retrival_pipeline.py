import os
from langchain_core.embeddings import embeddings
from langchain_core.tools import retriever
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma 

load_dotenv()

persist_directory='db/chroma_db'


#load embedding models and vector store
embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")


#recreate the vector store
db=Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hsnw:space":"cosine"}
)


# Search for relevant documents related to this query
# query="In what year did tesla begin production of the roadster"
query="On what date did Nvidia go public?"


#create a retriver obj
retriever = db.as_retriever(search_kwargs={"k":5})

relevant_docs=retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context---")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n {doc.page_content}\n")
