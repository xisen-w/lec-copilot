import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
from Agents.lectureAgent import LectureAgent
import re
import time

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

# Initialize session state
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'idea_cards' not in st.session_state:
    st.session_state.idea_cards = None
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'step' not in st.session_state:
    st.session_state.step = 1  # Track which step we're on

# Add log message
def add_log(message):
    st.session_state.logs.append(f"{time.strftime('%H:%M:%S')} - {message}")

# Show logs in an expander
with st.expander("Debug Logs", expanded=False):
    for log in st.session_state.logs:
        st.write(log)

# STEP 1: Record Audio
st.subheader("Step 1: Record Your Lecture")

# Always display the recorder
audio_data = audio_recorder(
    energy_threshold=(-1.0, 1.0),      # Always record
    pause_threshold=999999.0,          # Don't auto-stop
    sample_rate=44100,                 # CD quality
    text="Click to start/stop recording"
)

# If audio data is returned, save it
if audio_data:
    # Get byte length
    audio_size = len(audio_data)
    add_log(f"Audio recorded: {audio_size} bytes")
    
    # Check if audio is long enough
    if audio_size < 4000:
        st.error("Audio recording is too short. Please record at least 1-2 seconds.")
        add_log("Audio too short - rejected")
    else:
        # Save the audio data
        st.session_state.audio_bytes = audio_data
        add_log("Audio saved to session state")

# STEP 2: Process Audio if Available
if st.session_state.audio_bytes:
    st.subheader("Step 2: Review Recording")
    
    # Play the recorded audio
    st.audio(st.session_state.audio_bytes, format="audio/wav")
    
    # Show audio info
    audio_size = len(st.session_state.audio_bytes)
    st.write(f"Audio size: {audio_size} bytes")
    
    # Provide download button
    st.download_button(
        label="Download Recording",
        data=st.session_state.audio_bytes,
        file_name="lecture_recording.wav",
        mime="audio/wav"
    )
    
    # STEP 3: Transcribe
    if st.session_state.step == 1:  # Only show transcribe button in step 1
        if st.session_state.transcription is None:
            if st.button("Transcribe Recording"):
                add_log("Starting transcription process")
                
                # Create a temporary file for the audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(st.session_state.audio_bytes)
                    audio_path = tmp_file.name
                    add_log(f"Saved to temporary file: {audio_path}")
                
                try:
                    # Show a progress indicator
                    progress_bar = st.progress(50)
                    st.write("Transcribing audio...")
                    
                    # Transcribe the audio
                    transcription = lecture_agent.record_lecture(audio_path)
                    
                    if transcription and len(transcription) > 0:
                        add_log(f"Transcription successful: {len(transcription)} characters")
                        st.session_state.transcription = transcription
                        st.session_state.step = 2  # Move to next step
                    else:
                        add_log("Transcription failed - no text generated")
                        st.error("Transcription failed. No text was generated.")
                    
                    # Clean up temporary file
                    try:
                        os.unlink(audio_path)
                        add_log("Temporary file deleted")
                    except Exception as e:
                        add_log(f"Failed to delete temporary file: {str(e)}")
                    
                except Exception as e:
                    add_log(f"Error during transcription: {str(e)}")
                    st.error(f"An error occurred during transcription: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
    
    # Display transcription if available
    if st.session_state.transcription:
        st.subheader("Transcription")
        st.write(st.session_state.transcription)
        
        # STEP 4: Analyze
        if st.session_state.step == 2 and st.session_state.idea_cards is None:  # Only analyze in step 2
            if st.button("Analyze Content"):
                add_log("Starting content analysis")
                
                try:
                    # Show progress
                    progress_bar = st.progress(50)
                    st.write("Analyzing content...")
                    
                    # Analyze the transcription
                    idea_cards = lecture_agent.explain(st.session_state.transcription)
                    
                    if idea_cards and hasattr(idea_cards, 'idea_cards'):
                        add_log(f"Analysis complete: {len(idea_cards.idea_cards)} idea cards")
                        st.session_state.idea_cards = idea_cards
                        st.session_state.step = 3  # Move to final step
                    else:
                        add_log("Analysis failed - no idea cards generated")
                        st.error("Analysis failed. No idea cards were generated.")
                    
                except Exception as e:
                    add_log(f"Error during analysis: {str(e)}")
                    st.error(f"An error occurred during analysis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
        
        # Display analysis results if available
        if st.session_state.idea_cards and hasattr(st.session_state.idea_cards, 'idea_cards'):
            st.subheader("Key Ideas")
            for card in st.session_state.idea_cards.idea_cards:
                st.markdown(f"**{card.idea_name}**")
                render_latex(card.idea_explanation)
                
                with st.expander("Detailed Explanation"):
                    context_points = card.idea_context.split('\n')
                    for point in context_points:
                        if point.strip():
                            render_latex(f"- {point.strip()}")

# Reset button
if st.session_state.audio_bytes:
    if st.button("Record a New Lecture"):
        add_log("Resetting session")
        st.session_state.audio_bytes = None
        st.session_state.transcription = None
        st.session_state.idea_cards = None
        st.session_state.step = 1
        # Clear cache using the modern API
        st.cache_data.clear()
        # No need for experimental_singleton.clear() as it's deprecated
        st.rerun()
