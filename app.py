

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
def generate_questions(data):
    questions = []
    for row in data.itertuples():
        text = str(row[1])
        prompt = "Please formulate maximum 5 questions based on this data:\n" + text
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
            n=1,
            stop=None,
            timeout=30,
            )
        quest = response.choices[0].text.strip()
        questions.append(quest)
    return questions

def main():
    st.title('Tabular Data Key Points Generator')
    
    # Get input data from user
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Generate key points
        key_points = generate_key_points(data)
        questions = generate_questions(data)
        
        # Show key points
        st.subheader('Key Points')
        st.write(key_points)
        
        # Show questions
        st.subheader('Questions')
        st.write(questions)
        
if __name__ == '__main__':
    main()
