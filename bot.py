""" import openai
import gradio as gr
import os

openai.api_key = os.environ.get('AI')

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def chatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
    description="Ask anything you want",
    theme="compact").launch(share=True,
        server_name="0.0.0.0",
        server_port = 8080
        ) """

import os
OPENAI_API_KEY = os.getenv("AI")
if not OPENAI_API_KEY:
  raise "Env variable AI Key not specified"

#os.environ["OPENAI_API_KEY"] = '{my_api_key}'

import streamlit as st
from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from langchain import OpenAI

doc_path = './data/'
index_file = 'index.pdf'

if 'response' not in st.session_state:
    st.session_state.response = ''

def send_click():
    st.session_state.response  = index.query(st.session_state.prompt)

index = None
st.title("HAL")

sidebar_placeholder = st.sidebar.container()
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

    doc_files = os.listdir(doc_path)
    for doc_file in doc_files:
        os.remove(doc_path + doc_file)

    bytes_data = uploaded_file.read()
    with open(f"{doc_path}{uploaded_file.name}", 'wb') as f: 
        f.write(bytes_data)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(uploaded_file.name)
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    max_input_size = 1096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    index.save_to_disk(index_file)

""" elif os.path.exists(index_file):
    index = GPTVectorStoreIndex.load_index_from_storage(index_file)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    doc_filename = os.listdir(doc_path)[0]
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(doc_filename)
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')
 """
if index != None:
    st.text_input("Ask something: ", key='prompt')
    st.button("Send", on_click=send_click)
    if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")