from langchain.document_loaders import UnstructuredURLLoader
from scrape_utils import scrape
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings 
import streamlit as st 
import pinecone
import json 
import os
from llama_index import(
    GPTVectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    Document, 
)
from langchain import OpenAI
from langchain.docstore.document import Document as LangchainDocument
from llama_index.node_parser import SimpleNodeParser

import hashlib 


tokenizer = tiktoken.get_encoding("cl100k_base")

#Openapi Key 
OPENAI_API_KEY = os.getenv("AI")
if not OPENAI_API_KEY:
  raise "Env variable AI Key not specified"

documents = []

def load_documents_to_gpt_vectorstore(url):
  from llama_index import download_loader 

  urls = scrape(url)
  BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
  loader = BeautifulSoupWebReader()
  documents = loader.load_data(urls)
  parser = SimpleNodeParser()

  nodes = parser.get_nodes_from_documents(documents)
  llm_predictor = LLMPredictor(
    llm = OpenAI(
          temperature=0, model_name="text-davinci-003", OPENAI_API_KEY = os.getenv("AI")
        )
  )

  max_input_size = 4096
  num_output = 256
  max_chunk_overlap = 20
  prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

  service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper
  )
  index = GPTSimpleVectorIndex(nodes, service_context=service_context)
  index.save_to_disk("./gpt_index_docs_api_kube_v2.json")
  return index 

def chat(query):
  index = GPTSimpleVectorIndex.load_from_disk("gpt_index_docs.json")
  response = index.query(query)
  print(response)

st.title("Tiger Bot")

doc_input = st.text_input("URL I require")

if st.button("load documents"):
  st.markdown(load_documents_to_gpt_vectorstore(doc_input))
  st.success()

#question = st.text_input("Question you have:", key='prompt')

#if st.button("Answer you seek"):

  
if st.text_input("Ask something: ", key='prompt')
    st.button("Send", on_click=send_click)
    if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")
