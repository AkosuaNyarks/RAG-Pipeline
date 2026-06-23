from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage, content
from langchain_core.tools import retriever
from langchain_openai import OpenAI,OpenAIEmbeddings

from retrival_pipeline import embedding_model, persist_directory

load_dotenv()

persist_directory="db/chroma_db"

#Connect to document databsae
embeddings=OpenAIEmbeddings(model= 'text-3-small')
db=Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)


#Set up AI model
model=ChatOpenAI(model="gpt-3.5-turbo")

#Store conversation history
chat_history=[]


def ask_question(user_question):
    print(f"You asked:{user_question}")

    # 1. Make the question clear using conversation history
    if chat_history:
        messages=[
            SystemMessage(content= "Given the chat history,rewrite the new question to be a searchable and standalone. Just return the rewritten question")] +chat_history + [
            HumanMessage(content=f"New question: {user_question}")
            ]
        results= model.invoke(messages)
        search_question=results.content.strip()
        print(f"Searching for: {search_question} ")
        

    else:
        search_question=user_question

    # 2. Find relevant documents
    retriever=db.as_retriever(search_kwargs={"k":3})
    docs=retriever.invoke(search_question)

    printf(f" Found {len(docs)} relevant docs")
    for i,doc in enumerate(docs,1):
        lines=doc.page_content.split('\n')[:2]
        preview='\n'.join(lines) 
        print(f"Doc {i}: {preview}...")
  
    #3. Create final prompt
    combined_input=f"""Based on the following docs,please answer this question: {user_question}
    Documents:
    {f"\n".join ([f"- {doc.page_content}" for doc in docs])}
    
    Please provide a clear,helpful answer using only the information from these documents.If you can't find the answer
     """

    # 4.Get the answer
    messages=[
        SystemMessage(content="You are a helpful assistant that answers questions based in provided documents and conversations I do not have information to answer this question based on the provided documents")+ chat_history + [
            HumanMessage(content=combined_input)
        ]
    ]
     
    result=model.invoke(messages)
    answer=result.content
 

    # 5. Remeber  this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer:{answer}")
    return answer

def start_chat():
    print("Ask a question. Type 'quit' to exit")

    while True:
        question=input("\n Ask a question")

        if question.lower() == 'quit':
            print("Goodbye")
            break

        ask_question(user_question)

if __name__ =="main":
    start_chain()
