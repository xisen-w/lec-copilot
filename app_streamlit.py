import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
from Agents.lectureAgent import LectureAgent
import re

def render_latex(text):
    # Split the text into LaTeX and non-LaTeX parts
    parts = re.split(r'(\$\$?.*?\$\$?)', text)
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            # Display formula
            st.latex(part.strip('$'))
        elif part.startswith('$') and part.endswith('$'):
            # Inline formula
            st.markdown(part)
        else:
            # Regular text
            st.write(part)

# Initialize the LectureAgent
model_name = "gpt-4o-mini"  # Replace with your actual model name
lecture_agent = LectureAgent(model_name=model_name)

st.title("Lecture Transcription and Analysis")

# Create an instance of the audio recorder
audio_bytes = audio_recorder()

if audio_bytes:
    # Play the recorded audio
    st.audio(audio_bytes, format="audio/wav")

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        audio_path = tmp_file.name

    # Process the audio with LectureAgent
    try:
        st.write("Processing your lecture...")
        transcription = lecture_agent.record_lecture(audio_path)
        st.subheader("Transcription:")
        st.write(transcription)

        # Explain key ideas
        idea_cards_schema = lecture_agent.explain(transcription)
        st.subheader("Key Ideas:")
        for card in idea_cards_schema.idea_cards:
            st.markdown(f"**{card.idea_name}**")
            
            # Render the explanation with LaTeX support
            render_latex(card.idea_explanation)
            
            with st.expander("See detailed explanation"):
                # Split the context into bullet points
                context_points = card.idea_context.split('\n')
                for point in context_points:
                    if point.strip():  # Check if the point is not empty
                        render_latex(f"- {point.strip()}")

    except Exception as e:
        st.error(f"An error occurred while processing the audio: {str(e)}")

    # Clean up the temporary file
    os.unlink(audio_path)

    # Provide a download button for the recording
    st.download_button(
        label="Download Recording",
        data=audio_bytes,
        file_name="recording.wav",
        mime="audio/wav"
    )
else:
    st.info("Please record your lecture using the microphone above.")
