from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
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

    # Make the question clear using conversation history
    if chat_history:
        messages=[
            SystemMessage(content= "Given the chat history,rewrite the new question to be a searchable and standalone. Just return the rewritten question")] +chat_history + [
            HumanMessage(content=f"New question: {user_question}")
            ]
        results= model.invoke(messages)
        search_question=results.content.strip()
        

    else:
        search_question=user_question


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
