import requests
import streamlit as st


def get_llama_response(input):
    response = requests.post("http://localhost:8501/chatbot/invoke", json={'input': {'Questions': input}})
    return response.json()['output'] #change to from_messages if not working.

st.title('Burger club Chat Support')
input_text = st.text_input("Ask something:")



if input_text:
    st.write(get_llama_response(input_text))
