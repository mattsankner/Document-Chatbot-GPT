# Imports
from fastbook import *
from fastai.vision.widgets import *
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import gradio as gr
#import openai
import sys

print(sys.path)


os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY_HERE"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "good question!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
)

# Process a given URL and set up the QA chain
def setup_chain(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain

# Gradio Interface Function
def process_input(url, query):
    qa_chain = setup_chain(url)
    result = qa_chain({"query": query})
    return result["result"]

# Create the Gradio Interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.inputs.Textbox(placeholder="Enter a URL..."), gr.inputs.Textbox(placeholder="Ask a question...")],
    outputs="Answer",
    live=True,
    capture_session=True,
)

# Launch the Gradio Interface
iface.launch(share=True)

