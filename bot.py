import os
import streamlit as st
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext, OpenAIEmbedding
from llama_index import StorageContext, load_index_from_storage
from langchain import OpenAI
from fastapi import FastAPI

OPENAI_API_KEY = os.getenv("AI")
if not OPENAI_API_KEY:
  raise "Env variable AI Key not specified"

app = FastAPI()
doc_path = '/docs/'
index_file = 'index.pdf'
index = None

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
prompt_helper = PromptHelper(context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None)
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper,embed_model=embed_model)

def load_context():
    documents = SimpleDirectoryReader(doc_path).load_data()
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, prompt_helper=prompt_helper)
    index.storage_context.persist(persist_dir="<persist_dir>")
    return index 

index = load_context()

@app.get("/query")
def query(question: str):
    # storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
    # index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(service_context=service_context, verbose=True,response_mode="compact")
    return query_engine.query(question)
    st.session_state.response = query_engine.query(st.session_state.prompt)
