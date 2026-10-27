import os 
from dotenv import load_dotenv

load_dotenv()

openrouter_key=os.getenv("OPENROUTER_API_KEY")
cohere_key=os.getenv("COHERE_API_KEY")

from langchain_openrouter import ChatOpenRouter
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
import gradio as gr

def get_llm():
    model = 'openrouter/free'
    OpenRouter_llm = ChatOpenRouter(
        model=model,
        temperature=0.5,
        max_tokens=256,
        api_key=openrouter_key,
    )
    return OpenRouter_llm

def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def cohere_embedding():
    cohere_embedding = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        #truncate="END",
        cohere_api_key=cohere_key,
    )
    return cohere_embedding

def vector_database(chunks):
    embedding_model = cohere_embedding()
    vectordb = Chroma.from_documents(chunks , embedding_model)
    return vectordb

prompt=PromptTemplate(
    template="""
    Use only the context below to answear the question.
    Give only the final answer.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
    input_variables=["context","question"]

)

def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    chain_type_kwargs={"prompt":prompt},
                                    return_source_documents=False)
    response = qa.invoke({"query":query})
    return response['result']


with open("style.css","r",encoding="utf-8") as f:
    custom_css= f.read()

rag_application = gr.Interface(
    fn=retriever_qa,
    #allow_flagging=False,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath",elem_id="upload_pdf"),  
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...",elem_id="query_box")
    ],
    outputs=gr.Textbox(label="Answear",elem_id="answer_box"),
    title="ChatBot based on RAG and Langchain",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
    css=custom_css
)

rag_application.launch(server_name="127.0.0.1", server_port= 7860)