import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document  # Import the Document class
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import os
from dotenv import load_dotenv
load_dotenv()
from IPython.display import Markdown, display


groq_api_key = st.secrets["GROQ_API"]
st.set_page_config(page_title="LangChain: In-depth Summary from YT or Website", page_icon="🦜")
st.title("🦜 LangChain: In-depth Summary from YT or Website")
st.subheader('Explore and Summarize URL')

generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)


prompt_template = """
Perform an in-depth exploration of the content from the following source, covering the main topics, unique features, and key insights:
- Start with an overview of the primary themes.
- Identify and summarize the key sections or main points in a structured manner.
- Include any significant details or unique aspects that make this content valuable.
- Aim to provide a comprehensive summary in 300 words.

Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Function to get YouTube transcript using youtube-transcript-api and format it with TextFormatter
def get_youtube_transcript(url):
    try:
        video_id = url.split("v=")[-1] if "youtube.com" in url else url.split("/")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format transcript using TextFormatter
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)
        
        # Create a Document instance from the transcript text
        document = Document(page_content=transcript_text, metadata={"source": "YouTube"})
        
        return document
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                docs = None  # Initialize docs to None
                
                ## Loading the website or YouTube video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    transcript_document = get_youtube_transcript(generic_url)
                    if transcript_document:
                        docs = [transcript_document]  # List of Document objects
                    else:
                        st.error("No transcript available. Unable to retrieve content from YouTube.")
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()

                # Only proceed if docs is defined
                if docs:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    st.markdown(output_summary)
                else:
                    st.error("Failed to load content from the provided URL.")
                    
        except Exception as e:
            st.exception(f"Exception: {e}")
