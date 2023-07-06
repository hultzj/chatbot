import os
import requests
import streamlit as st

st.title("Roku Bot")

if 'response' not in st.session_state:
    st.session_state.response = ''

def query():
    data = requests.get("http://roku.rrama.svc.cluster.local:8080/query?question="+st.session_state.prompt).json()
    st.session_state.response = data['response']

st.text_input("Ask something: ", key='prompt')
st.button("Send", on_click=query)
if st.session_state.response:
    st.subheader("Response: ")
    st.success(st.session_state.response, icon= "🤖")