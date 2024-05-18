import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import bs4
import time
from langserve import add_routes  
from fastapi import FastAPI
import uvicorn


from dotenv import load_dotenv
load_dotenv()

## load the Chatgpt/groq API key and setting langsmith tracking.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.environ['GROQ_API_KEY']


embeddings=OllamaEmbeddings()
#st.session_state.embeddings= OpenAIEmbeddings()

loader=WebBaseLoader(web_paths=("https://theburgerclub.in/about",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("common-para")
                     )))# fetching all about burger club from official website.
    
loader=WebBaseLoader(web_paths=("https://theburgerclub.in/store-locator",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("font-weight-light","mt-2")
                     )))
    
loader=WebBaseLoader(web_paths=("https://theburgerclub.in/order/the-burger-club-rani-bagh-delhi",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("wla-outlet-name-md","item-title","heading-customize more30857606","price-p")
                     )))
docs=loader.load()
# Ensure embeddings are generated before using FAISS
embeddings = OllamaEmbeddings()
docs = loader.load()  # Assuming loader is defined correctly
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])

# Check if there are any documents and embeddings before proceeding
if final_documents and embeddings:
    vectors = FAISS.from_documents(final_documents, embeddings)
    # ... rest of your code using vectors
else:
    print("Error: No documents or embeddings found. Check website loading and embedding process.")


#st.title("BurgerClub Bot")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-8b-8192")
#llm=ChatOpenAI(model="gpt-3.5-turbo")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
You have to act as an official chatbot for "The burger club" in india but you only answer from the context provided to you.
Please provide the most accurate response based on the question
"Based on the provided context" dont repeat this in every response
 just try to be a real chatbot for this company and answer only if you know something 
 Refuse polietely if you are not 100 percent certain about the info , dont hallucinate. 
 tell to visit official website:www.theburgerclub.in for further info after each responce.
 Once again only answer questions from context provide , you wont answer from your own learnings. 
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

app = FastAPI(
    title="Langchain Server",
    version = "1.0",
    description="API server for burger_club_chat"
)
add_routes(
    app,
    retrieval_chain,
    path="/chatbot"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8501)

'''prompt=st.text_input("Ask your question?:")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Some more context"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.'''