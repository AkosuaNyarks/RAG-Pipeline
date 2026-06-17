import os
from langchain_core.embeddings import embeddings
from langchain_core.tools import retriever
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage


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

# Combine query and relevant chunks to generate a response from the LLM
combined_input= f"""Based on the following documents, answer this question:{query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}
Provide a clear, helpful answer using only information from these documents.If the answer is not explicitly stated in the provided documents,
do not infer or guess, say " I do not have information to answer this question based on the provided documents"
"""

#Create an Open AI model
model=ChatOpenAI(model="gpt-3.5-turbo")

#Define the message/OPen AI API call
messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input),
]

#Invoke the model with the message
result=model.invoke(messages)
print("\n ---Generated Response ---")
print("Content Only:")
print(result.content)