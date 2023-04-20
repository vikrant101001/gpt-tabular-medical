

import streamlit as st
import pandas as pd
import openai
import os

st.set_page_config(layout="wide",page_title="GPT Tabular Clinical Webapp")


openai.api_key = os.getenv('OPENAI_KEY')

# Define function to generate key points using GPT-3
def generate_key_points(data):
    key_points = []
    for row in data.itertuples():
        text = str(row[1])
        prompt = "Please summarize the following text in a few key points with numbering:\n" + text
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
            n=1,
            stop=None,
            timeout=30,
            )
        summary = response.choices[0].text.strip()
        key_points.append(summary)
    return key_points

# Define function to generate questions using GPT-3
def generate_qa(key_points):
    qa_pairs = []
    for kp in key_points:
        prompt = "Please generate a question and answer based on the following key point:\n" + kp
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
            n=1,
            stop=None,
            timeout=30,
            )
        qa = response.choices[0].text.strip().split("\n")
        qa_pairs.append((qa[0], qa[1]))
    return qa_pairs

def main():
    st.title('Tabular Data Key Points Generator')
    
    # Get input data from user
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Generate key points
        key_points = generate_key_points(data)
        
        
        # Show key points
        st.subheader('Key Points')
        st.write(key_points)
        
        # Generate questions and answers
        qa_pairs = generate_qa(key_points)
        
        # Show questions and answers
        st.subheader('Questions and Answers')
        for i, qa in enumerate(qa_pairs):
            st.write(f"{i+1}. Q: {qa[0]}")
            st.write(f"   A: {qa[1]}")

        
if __name__ == '__main__':
    main()
