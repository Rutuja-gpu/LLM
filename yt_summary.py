import os
import asyncio
import streamlit as st
import yt_dlp
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time

# Disable Streamlit file watcher for torch
os.environ["STREAMLIT_WATCH_FILE"] = "false"

# Ensure asyncio event loop is initialized
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(layout="wide")

def download_video(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'downloads/%(title)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        st.error(f"An error occurred while downloading the video: {e}")
        return None

def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    try:
        whisper = WhisperTranscriber()
        st.write("WhisperTranscriber initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing WhisperTranscriber: {e}")
        return None

    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    st.write("Pipeline output:", output)
    return output

def main():
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with LLaMA 2 🦙, Haystack, Streamlit and ❤️')

    with st.expander("About the App"):
        st.write("This app allows you to summarize YouTube videos with LLaMA and Whisper.")
    
    youtube_url = st.text_input("Enter YouTube URL")

    if st.button("Submit") and youtube_url:
        start_time = time.time()
        file_path = download_video(youtube_url)

        if file_path:
            full_path = "/Users/rutu/Downloads/llama-2-7b-32k-instruct.Q3_K_S.gguf"
            model = initialize_model(full_path)
            prompt_node = initialize_prompt_node(model)

            output = transcribe_audio(file_path, prompt_node)

            end_time = time.time()
            elapsed_time = end_time - start_time

            col1, col2 = st.columns([1, 1])
            with col1:
                st.video(youtube_url)
            with col2:
                st.header("Summarization of YouTube Video")
                if output and "results" in output and output["results"]:
                    st.success(output["results"][0].split("\n\n[INST]")[0])
                else:
                    st.error("No summarization results found.")
                st.write(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()