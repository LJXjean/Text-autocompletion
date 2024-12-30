# streamlit run frontend_demo.py
# for running the frontend demo

import streamlit as st
import requests
from streamlit.components.v1 import html

st.title("Text Autocompletion Demo")

# Initialize session state for storing completion
if 'completion' not in st.session_state:
    st.session_state.completion = ''
if 'accepted_text' not in st.session_state:
    st.session_state.accepted_text = ''

def on_text_change():
    # Clear completion when text changes
    st.session_state.completion = ''

text_area_container = st.empty()
user_text = text_area_container.text_area(
    "Type your text here (context)...", 
    height=150,
    key="user_input",
    on_change=on_text_change
)

# Custom CSS and JavaScript for handling completion acceptance
st.markdown("""
<style>
.suggestion {
    color: #808080;
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# Update text area if completion was accepted
if st.session_state.accepted_text:
    user_text = st.session_state.accepted_text
    text_area_container.text_area(
        "Type your text here (context)...",
        value=st.session_state.accepted_text,
        height=150,
        key="user_input_updated"
    )
    st.session_state.accepted_text = ''

if st.button("Get Suggestion") or st.session_state.completion:
    if not st.session_state.completion:  # Only make API call if no completion exists
        payload = {
            "text_before_cursor": user_text,
            "max_length": 50
        }
        response = requests.post("http://127.0.0.1:8000/autocomplete", json=payload)
        if response.status_code == 200:
            st.session_state.completion = response.json()["completion"]
        else:
            st.error(f"Error: {response.status_code}")
            st.session_state.completion = ''

    # Display text with suggestion
    if st.session_state.completion:
        col1, col2 = st.columns([4,1])
        with col1:
            st.markdown(
                f"{user_text}<span class='suggestion'>{st.session_state.completion}</span>",
                unsafe_allow_html=True
            )
        with col2:
            if st.button("Accept"):
                new_text = user_text + st.session_state.completion
                st.session_state.accepted_text = new_text
                st.session_state.completion = ''
                # Update the text area with accepted completion
                st.rerun()
