import os

OPENAI_API_KEY = os.getenv("AI")
if not OPENAI_API_KEY:
  raise "Env variable AI Key not specified"

import streamlit as st
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext, OpenAIEmbedding
from llama_index import StorageContext, load_index_from_storage
from langchain import OpenAI

doc_path = '/docs/'
index_file = 'index.pdf'
index = None
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
prompt_helper = PromptHelper(context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None)
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper,embed_model=embed_model)

if 'response' not in st.session_state:
    st.session_state.response = ''

def send_click():
    storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
    
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(service_context=service_context, verbose=True,response_mode="compact")

    st.session_state.response  = query_engine.query(st.session_state.prompt)

def load_context():

    st.title("HAL")

    sidebar_placeholder = st.sidebar.container()

    documents = SimpleDirectoryReader(doc_path).load_data()
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(index_file)
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

    documents = SimpleDirectoryReader(index_file).load_data()

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, prompt_helper=prompt_helper)

    index.storage_context.persist(persist_dir="<persist_dir>")

    return index 

index = load_context()

if index != None:
    st.text_input("Ask something: ", key='prompt')
    st.button("Send", on_click=send_click)
    if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")