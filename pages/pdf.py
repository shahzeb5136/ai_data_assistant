import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
import requests
import json
from openai import OpenAI
import traceback

# --- Configuration (Ideally, centralize this or pass clients/configs) ---
# You might want to pass the OpenAI client and Ollama config
# from your main app or load them here securely.
# For demonstration, we assume access similar to your SQL functions.

OLLAMA_URL = "http://localhost:11434/api/chat" # Reuse from main app config
MODEL_NAME = "llama3.1:latest" # Reuse from main app config

# Make sure the OpenAI client is initialized somewhere accessible
# Example: client = OpenAI(api_key=st.secrets["openai"]["api_key"])
# Using a placeholder here - ensure your actual client is used
try:
    # Attempt to reuse the client if already initialized in the session state
    # This depends on how you initialize it in your main app or structure
    if "openai_client" in st.session_state:
        client = st.session_state.openai_client
    else:
        # Fallback: Initialize here (ensure API key is handled securely)
        # Replace with your actual API key retrieval method
        client = OpenAI(api_key="YOUR_OPENAI_API_KEY") # Replace!
        st.session_state.openai_client = client # Optional: store for reuse
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    client = None


# System Prompt for PDF Q&A
PDF_SYSTEM_PROMPT = """
You are an AI assistant specialized in answering questions based *only* on the provided text content from a PDF document.
Do not use any prior knowledge or external information.
If the answer cannot be found within the provided text, state that clearly.
Be concise and directly answer the question using information from the text.

Provided Text:
{pdf_content}
"""


# --- Helper Functions ---
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_bytes = uploaded_file.getvalue()
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # Ensure text was extracted
                 text += page_text + "\n" # Add newline between pages
        if not text:
            st.warning("Could not extract text from the PDF. The PDF might be image-based or corrupted.", icon="⚠️")
            return None
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        st.error(traceback.format_exc()) # More detailed error for debugging
        return None

def get_answer_from_pdf_text_ollama(pdf_text, user_question):
    """Sends PDF text and question to local Ollama and returns the answer."""
    if not pdf_text:
        return "Error: Could not process PDF text."

    full_prompt = PDF_SYSTEM_PROMPT.format(pdf_content=pdf_text)

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user_question}
                ],
                "stream": False # Get the full response at once
            })
        )
        response.raise_for_status() # Raise an exception for bad status codes

        response_data = response.json()
        # Extract the message content from the response structure
        if "message" in response_data and "content" in response_data["message"]:
             return response_data["message"]["content"].strip()
        else:
            st.error(f"Unexpected Ollama response format: {response_data}")
            return "Error: Could not parse response from Ollama."

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Ollama: {e}")
        st.error("Please ensure the Ollama server is running and accessible.")
        return "Error: Could not connect to the local Ollama model."
    except Exception as e:
        st.error(f"An error occurred while querying Ollama: {e}")
        st.error(traceback.format_exc())
        return "Error: An unexpected error occurred."


def get_answer_from_pdf_text_openai(pdf_text, user_question):
    """Sends PDF text and question to OpenAI API and returns the answer."""
    if not pdf_text:
        return "Error: Could not process PDF text."
    if not client:
         return "Error: OpenAI client not initialized."

    full_prompt = PDF_SYSTEM_PROMPT.format(pdf_content=pdf_text)

    try:
        # Using the ChatCompletion endpoint
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or specify another model like gpt-4
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": user_question}
            ]
        )
        # Extract the response content correctly
        if completion.choices and completion.choices[0].message:
            return completion.choices[0].message.content.strip()
        else:
            st.error(f"Unexpected OpenAI response format: {completion}")
            return "Error: Could not parse response from OpenAI."

    except Exception as e:
        st.error(f"An error occurred while querying OpenAI: {e}")
        st.error(traceback.format_exc())
        return "Error: An unexpected error occurred while querying OpenAI."
    

# --- App layout ---
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

# --- Page Configuration ---
st.set_page_config(page_title="PDF Query Assistant", layout="wide") # Config might be set in main app.py

# --- Session State Initialization ---
# Ensure necessary session state variables exist for this page
if "pdf_text_content" not in st.session_state:
    st.session_state.pdf_text_content = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "pdf_answer" not in st.session_state:
    st.session_state.pdf_answer = None
if "pdf_question" not in st.session_state:
     st.session_state.pdf_question = ""

# --- Sidebar ---
# Reuse sidebar elements if defined in a central app.py or duplicate if needed per page
with st.sidebar:
     # LLM Selection (reuse or adapt from your SQL page structure)
     # Ensure llm_choice is accessible, perhaps set in a main app.py or initialized here if needed
    if "llm_choice" not in st.session_state:
        st.session_state.llm_choice = "Chat GPT" # Default

    st.session_state.llm_choice = st.selectbox(
        "Select Model for PDF Q&A",
        ["Chat GPT", "Local Ollama"],
        index=0 if st.session_state.llm_choice == "Chat GPT" else 1,
        key="pdf_llm_select" # Use a unique key if sharing session state
    )
    st.info(f"Using {st.session_state.llm_choice} for PDF questions.")
    if st.session_state.llm_choice == "Local Ollama":
        st.warning("Ensure the Ollama server is running.", icon="🔌")

# --- Main Content ---
st.title("📄 PDF Query Assistant")
st.markdown(
    "Upload a PDF document and ask questions about its content. "
    "The AI will answer based *only* on the information within the document."
)

# --- PDF Upload ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Check if it's a new file or the same one
    if st.session_state.pdf_filename != uploaded_file.name:
        st.session_state.pdf_answer = None # Clear previous answer
        st.session_state.pdf_question = "" # Clear previous question
        st.session_state.pdf_filename = uploaded_file.name
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            st.session_state.pdf_text_content = extract_text_from_pdf(uploaded_file)
            if st.session_state.pdf_text_content:
                st.success(f"✅ Successfully processed '{uploaded_file.name}'. Ready for questions.")
            else:
                st.error("Failed to extract text. Please try a different PDF.")
                st.session_state.pdf_filename = None # Reset filename if processing failed
    else:
        # If the same file is uploaded or remains, indicate it's ready
        if st.session_state.pdf_text_content:
            st.info(f"Using previously uploaded file: '{st.session_state.pdf_filename}'. Ask your question below.", icon="ℹ️")
        # Handle case where previous upload of same name failed
        elif st.session_state.pdf_filename:
             st.warning(f"Previous attempt to process '{st.session_state.pdf_filename}' failed. Please try re-uploading or use a different file.", icon="⚠️")


# --- Question Input ---
if st.session_state.pdf_text_content:
    st.subheader("Ask a Question About the PDF")
    user_question_pdf = st.text_area(
        "Your question:",
        key="pdf_user_question_input",
        placeholder="e.g., What is the main conclusion mentioned in the document?",
        value=st.session_state.pdf_question # Retain question if needed
    )

    ask_button = st.button("💬 Ask Question", key="ask_pdf_button")

    if ask_button and user_question_pdf:
        st.session_state.pdf_question = user_question_pdf # Store question
        with st.spinner(f"Getting answer using {st.session_state.llm_choice}..."):
            if st.session_state.llm_choice == "Local Ollama":
                st.session_state.pdf_answer = get_answer_from_pdf_text_ollama(
                    st.session_state.pdf_text_content,
                    user_question_pdf
                )
            else: # Assuming Chat GPT (OpenAI)
                 st.session_state.pdf_answer = get_answer_from_pdf_text_openai(
                    st.session_state.pdf_text_content,
                    user_question_pdf
                 )

# --- Display Answer ---
if st.session_state.pdf_answer:
    st.subheader("💡 Answer")
    st.markdown(st.session_state.pdf_answer)

# Optional: Display extracted text in an expander for debugging/verification
if st.session_state.pdf_text_content:
     with st.expander("View Extracted Text (for verification)"):
         st.text_area("Extracted Content:", st.session_state.pdf_text_content, height=300, disabled=True)