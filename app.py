import streamlit as st
from dotenv import load_dotenv

load_dotenv() ##load all the nevironment variables
import os
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) ##configure the google generative ai with the api key

prompt = """You are a YouTube video summarizer. You will take the transcript text
and summarize the entire video, providing the important points in a bullet-point format
within 250 words. Please provide the summary of the text given here: """

## getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        print(video_id)
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e
    
## getting the summary based on Prompt from Google Gemini Pro
def generate_gemini_content(transcript_text,prompt):

    models = genai.list_models()
    for model in models:
        print(f"Model Name: {model.name}, Supported Methods: {model.supported_generation_methods}")

    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")  # Replace with the correct model name
        response = model.generate_content(prompt + transcript_text)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return "Failed to generate content. Please check the model configuration."

st.title("YouTube Transcript to Detailed Notes Converter")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Get Detailed Notes"):
    transcript_text=extract_transcript_details(youtube_link)

    if transcript_text:

        summary=generate_gemini_content(transcript_text,prompt)
        st.markdown("## Detailed Notes:")
        st.write(summary)



