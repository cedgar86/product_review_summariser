import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"  # Replace with your actual API key

# Initialize the chat-based model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Create a ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(
    "Please provide a concise summary of the following focus group discussion transcript:\n\n{transcript}"
)

# Define the runnable sequence for summarization
summarize_sequence = prompt_template | llm

# Streamlit app
st.title("Focus Group Transcript Summarizer")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing 'product' and 'transcript' columns", type="csv")

if uploaded_file:
    # Load the CSV file into a dataframe
    transcripts_df = pd.read_csv(uploaded_file)
    
    # Display the dataframe to the user (optional)
    st.write("Data Preview:", transcripts_df.head())
    
    # Ensure the CSV has the necessary columns
    if 'product' in transcripts_df.columns and 'transcript' in transcripts_df.columns:
        # Allow the user to select a product
        product_name = st.selectbox("Choose a product to summarize:", transcripts_df['product'].unique())
        
        if product_name:
            # Find the transcript for the selected product
            transcript_text = transcripts_df[transcripts_df['product'] == product_name]['transcript'].values[0]
            
            # Run the sequence to get the summary
            if st.button("Summarize"):
                with st.spinner("Generating summary..."):
                    response = summarize_sequence.invoke({"transcript": transcript_text})
                    summary_text = response.content
                    st.subheader(f"Summary for {product_name.capitalize()}")
                    st.write(summary_text)
    else:
        st.error("Uploaded file does not contain the required 'product' and 'transcript' columns.")