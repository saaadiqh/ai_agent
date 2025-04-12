import dotenv
dotenv.load_dotenv()
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

st.title("Summariser")

def get_text():
    input_text = st.text_input('Type in URL link below: '
                               , key='url_input'
                               )
    return input_text

user_input = get_text()

if user_input:
    # Load data
    loader = WebBaseLoader(user_input)
    data = loader.load()
    # Split data loaded into chunks
    # chunk size needs to be less than total max token size of LLM
    # GPT3.5 = 4096 max tokens.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500
                                                   , chunk_overlap = 0
                                                   )
    all_splits = text_splitter.split_documents(data)
    # Store data onto a vector space (vector store)
    vectorstore = Chroma.from_documents(documents=all_splits
                                        , embedding=OpenAIEmbeddings())
    # Prompt
    question = "Summarise the key points of this article"

    # Perform similarity search based on this question 
    # f we pass in a query, the vectorstore will embed the query, 
    # perform a similarity search over the embedded documents, 
    # and return the most similar ones. 
    docs = vectorstore.similarity_search(question)
    print(f"len of docs: {len(docs)}")

    # Retrieval QA is a type of LLM chain structure allowing you 
    # to ask questions about data and get an answer.
    # Copy pasted from LangChain RAG Prompt "Returning sources" section in link
    template = """Use the following pieces of context to answer the question at 
    the end. If you don't know the answer, just say that you don't know, 
    don't try to make up an answer. Use three sentences maximum and keep 
    the answer as concise as possible. Always say "thanks for asking!" 
    at the end of the answer.
    {context} 
    Question: {question}
    Helpful Answer: """
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Using chatopenai instead of openai, she finds it cheaper but not verified
    llm = ChatOpenAI(model_name="gpt-3.5-turbo"
                     , temperature=0
                     )
    
    # qa_chain uses prompt given and is an agent that will answer questions
    # based on all the data stored in the vectorstore of the data
    qa_chain = RetrievalQA.from_chain_type(
        llm
        , retriever = vectorstore.as_retriever()
        , chain_type_kwargs={'prompt': custom_rag_prompt }
    )

if user_input:
    result = qa_chain({"query": question})
    result["result"]