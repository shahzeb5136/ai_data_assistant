import streamlit as st
import requests
import json
import re
import pandas as pd
import psycopg2
from io import BytesIO
import traceback
from openai import OpenAI
import datetime
import openpyxl

# --- Configuration ---

# Ollama server URL and model name
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:latest"

# OpenAI API configuration (replace with secure handling in production)
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Database Credentials (Consider using Streamlit Secrets or Environment Variables for production)
DB_NAME = "test1"
DB_USER = "postgres"
DB_PASSWORD = "753951"
DB_HOST = "localhost"
DB_PORT = "5432"

# System Prompt for the LLM (for local Ollama)
SYSTEM_PROMPT = """
You are an AI agent that is meant to create a SQL query based on the users natural language question.
The table you will be querying will be as follows:

Schema: testschema
Tablename: encounters

Column Name             Data Type          Description
Id                      VARCHAR(50)        A unique identifier for the specific encounter record. This ID typically refers to a single visit or episode of care. This is the PRIMARY KEY
START                   TIMESTAMP          The date and time when the encounter began (e.g., patient check-in or admission time).
STOP                    TIMESTAMP          The date and time when the encounter ended (e.g., discharge or check-out time).
PATIENT                 VARCHAR(50)        A unique identifier for the patient involved in the encounter. This may be a de-identified or system-generated value.
ORGANIZATION            VARCHAR(100)       The name or identifier of the healthcare provider organization (e.g., hospital, clinic) where the encounter took place.
PAYER                   VARCHAR(100)       The insurance company or entity responsible for paying for the healthcare services during this encounter.
ENCOUNTERCLASS          VARCHAR(50)        The classification or type of encounter (e.g., "inpatient", "outpatient", "emergency", etc.). Helps categorize the setting or nature of care.
CODE                    VARCHAR(50)        A standardized medical or administrative code representing the encounter, usually from a system like CPT, HCPCS, or SNOMED.
DESCRIPTION             TEXT               A human-readable description of the encounter code, summarizing the nature of the encounter or services provided.
BASE_ENCOUNTER_COST     NUMERIC(10,2)      The standard cost of the encounter before adjustments or claims. Represents a baseline value for the care episode.
TOTAL_CLAIM_COST        NUMERIC(10,2)      The total amount billed or claimed for the encounter, including additional services, procedures, or adjustments.
PAYER_COVERAGE          NUMERIC(10,2)      The portion of the total claim cost covered by the payer (e.g., insurance). This may not include patient out-of-pocket responsibility.
REASONCODE              VARCHAR(50)        A medical code indicating the reason for the encounter, typically from standardized coding systems (e.g., ICD-10).
REASONDESCRIPTION       TEXT               A descriptive explanation of the reason for the encounter, such as a diagnosis or presenting complaint.

Remember it is very very important that you only return a SQL query. Nothing else. The query will be fed into SQL to generate a table.
So make very sure you only return a SQL query. I cannot stress this enough. No other text explaining what the query is doing should be included.
We are using PostgreSQL. Make sure to add the schema name in the SQL query as well (e.g., testschema.encounters).
Ensure the SQL query is valid PostgreSQL syntax.
Only select columns that are relevant to the user's question unless they ask for all columns or specific ones.
Add a semicolon at the end of the query.
"""

# System Prompt for the OpenAI API version
SYSTEM_PROMPT_OPENAI = SYSTEM_PROMPT

SYSTEM_PROMPT_QUERY_SUMMARIZER = """
You are an AI assistant. You will receive a PostgreSQL query.
Your task is to summarize what the query does in simple, non-technical language, suitable for a business user or executive.
Focus on the goal of the query (e.g., "This query finds...", "It calculates...", "It lists...") and the main conditions used.
Keep the summary concise, typically 1-3 sentences.

At the end of the summary also add in the definitions of the columns returned based on the below original data description.

Column Name             Data Type          Description
Id                      VARCHAR(50)        A unique identifier for the specific encounter record. This ID typically refers to a single visit or episode of care. This is the PRIMARY KEY
START                   TIMESTAMP          The date and time when the encounter began (e.g., patient check-in or admission time).
STOP                    TIMESTAMP          The date and time when the encounter ended (e.g., discharge or check-out time).
PATIENT                 VARCHAR(50)        A unique identifier for the patient involved in the encounter. This may be a de-identified or system-generated value.
ORGANIZATION            VARCHAR(100)       The name or identifier of the healthcare provider organization (e.g., hospital, clinic) where the encounter took place.
PAYER                   VARCHAR(100)       The insurance company or entity responsible for paying for the healthcare services during this encounter.
ENCOUNTERCLASS          VARCHAR(50)        The classification or type of encounter (e.g., "inpatient", "outpatient", "emergency", etc.). Helps categorize the setting or nature of care.
CODE                    VARCHAR(50)        A standardized medical or administrative code representing the encounter, usually from a system like CPT, HCPCS, or SNOMED.
DESCRIPTION             TEXT               A human-readable description of the encounter code, summarizing the nature of the encounter or services provided.
BASE_ENCOUNTER_COST     NUMERIC(10,2)      The standard cost of the encounter before adjustments or claims. Represents a baseline value for the care episode.
TOTAL_CLAIM_COST        NUMERIC(10,2)      The total amount billed or claimed for the encounter, including additional services, procedures, or adjustments.
PAYER_COVERAGE          NUMERIC(10,2)      The portion of the total claim cost covered by the payer (e.g., insurance). This may not include patient out-of-pocket responsibility.
REASONCODE              VARCHAR(50)        A medical code indicating the reason for the encounter, typically from standardized coding systems (e.g., ICD-10).
REASONDESCRIPTION       TEXT               A descriptive explanation of the reason for the encounter, such as a diagnosis or presenting complaint.
"""


# --- Helper Functions ---
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


# @st.cache_data(ttl=600) # Caching LLM calls is good
def generate_sql_query_ollama(user_question):
    """Sends the user question to Ollama and returns the generated SQL query."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0 # Low temp for deterministic SQL generation
        }
    }
    raw_response = "" # Initialize raw_response
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        raw_response = data.get("message", {}).get("content", "")

        # More robust cleaning
        clean_response = re.sub(r"```sql\s*", "", raw_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r"\s*```$", "", clean_response).strip()
        clean_response = re.sub(r"<think>.*?</think>", "", clean_response, flags=re.DOTALL).strip()
        clean_response = clean_response.replace(";", "").strip() # Remove existing semicolons before adding one

        if not clean_response:
             raise ValueError("Ollama returned an empty response.")

        # Basic check if it looks like SQL
        if not re.match(r"^\s*(SELECT|WITH)\s+", clean_response, re.IGNORECASE):
            raise ValueError(f"Generated text does not look like a valid SQL query: {clean_response}")

        clean_response += ';' # Ensure it ends with a semicolon
        return clean_response

    except requests.exceptions.Timeout:
        st.error(f"Error: Connection to Ollama timed out after 60 seconds. Is the server running at {OLLAMA_URL}?")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to Ollama server at {OLLAMA_URL}. Please ensure it's running and accessible.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Ollama API: {e}")
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        st.error(f"Error processing response from Ollama: {e}")
        st.error(f"Raw response was: {raw_response}") # Show raw response for debugging
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Ollama query generation: {e}")
        st.error(traceback.format_exc())
        return None

# @st.cache_data(ttl=600) # Caching LLM calls is good
def generate_sql_query_openai(user_question):
    """Sends the user question to OpenAI API and returns the generated SQL query."""
    if not client:
        st.error("OpenAI client is not initialized. Cannot generate query.")
        return None
    raw_response = "" # Initialize raw_response
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Use a powerful model like gpt-4o
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_OPENAI},
                {"role": "user", "content": user_question}
            ],
            temperature=0.0, # Low temp for deterministic SQL generation
            stream=False
        )
        raw_response = response.choices[0].message.content.strip()

        # More robust cleaning
        clean_response = re.sub(r"```sql\s*", "", raw_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r"\s*```$", "", clean_response).strip()
        clean_response = re.sub(r"<think>.*?</think>", "", clean_response, flags=re.DOTALL).strip()
        clean_response = clean_response.replace(";", "").strip() # Remove existing semicolons before adding one

        if not clean_response:
             raise ValueError("OpenAI returned an empty response.")

        # Basic check if it looks like SQL
        if not re.match(r"^\s*(SELECT|WITH)\s+", clean_response, re.IGNORECASE):
             raise ValueError(f"Generated text does not look like a valid SQL query: {clean_response}")

        clean_response += ';' # Ensure it ends with a semicolon
        return clean_response

    except Exception as e:
        st.error(f"Error generating SQL query with OpenAI: {e}")
        st.error(f"Raw response was: {raw_response}") # Show raw response for debugging
        st.error(traceback.format_exc())
        return None

# @st.cache_data(ttl=600) # Caching LLM calls is good
def summarize_sql_query(query_text):
    """Sends the SQL query to OpenAI to get a human-readable summary."""
    if not client:
        st.error("OpenAI client is not initialized. Cannot summarize query.")
        return None
    if not query_text:
        return "No query was generated to summarize."

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Use a capable model for summarization
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QUERY_SUMMARIZER},
                {"role": "user", "content": f"Summarize this SQL query:\n\n```sql\n{query_text}\n```"}
            ],
            temperature=0.5, # Allow some creativity for summarization
            stream=False
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"Error summarizing SQL query with OpenAI: {e}")
        st.error(traceback.format_exc())
        # Return a placeholder error message instead of None
        return "Error: Could not generate summary."

# Note: Do NOT cache run_sql_query with st.cache_data, as the underlying data might change.
def run_sql_query(sql_query):
    """Connects to the PostgreSQL database, runs the query, and returns a DataFrame.
    Limits results to 10,000 rows.
    """
    conn = None
    cursor = None
    MAX_ROWS = 20001 # Fetch one more than limit to check if exceeded

    if not sql_query:
        st.error("Cannot run query: No SQL query provided.")
        return None

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            connect_timeout=10 # Add a connection timeout
        )
        cursor = conn.cursor()

        # Add a statement timeout to prevent runaway queries (e.g., 60 seconds)
        cursor.execute("SET statement_timeout = '60s';")
        cursor.execute(sql_query)

        if cursor.description:  # Check if the query returned columns (e.g., SELECT)
            colnames = [desc[0] for desc in cursor.description]
            results = cursor.fetchmany(MAX_ROWS) # Fetch up to MAX_ROWS

            if len(results) == MAX_ROWS:
                st.error(f"Query returned more than {MAX_ROWS-1} rows. Please refine your query to be more specific.")
                # Return the first MAX_ROWS-1 results to show a preview, or return None
                # df = pd.DataFrame(results[:-1], columns=colnames) # Option 1: Return partial results
                return None # Option 2: Return None if limit exceeded
            else:
                df = pd.DataFrame(results, columns=colnames)
                return df
        else:
            # For non-SELECT queries (INSERT, UPDATE, DELETE)
            conn.commit()
            status_message = f"Query executed successfully. Rows affected: {cursor.rowcount}"
            st.success(status_message) # Use success message for non-SELECT
            # Return an empty DataFrame or status message DataFrame
            # return pd.DataFrame() # Empty DF might be better
            return pd.DataFrame({"Status": [status_message]})

    except psycopg2.errors.QueryCanceled as e: # Catch statement timeout specifically
        st.error(f"Database Error: Query timed out after 60 seconds. Please refine your query or check database performance.")
        return None
    except psycopg2.Error as e:
        st.error(f"Database Error: {e.pgcode} - {e.pgerror}")
        st.error(f"Query causing error: \n```sql\n{sql_query}\n```") # Show the problematic query
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during database query: {e}")
        st.error(traceback.format_exc())
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- Excel Function ---
def to_excel(results_df, user_question, query_summary, generated_sql): # Added generated_sql parameter
    """Converts results and query info to an Excel file in memory with two sheets."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Results
        if results_df is not None:
            # Check if it's the status DataFrame or actual results
            if "Status" in results_df.columns and len(results_df) == 1:
                 pd.DataFrame({"Message": [results_df["Status"].iloc[0]]}).to_excel(writer, index=False, sheet_name='Results')
            elif results_df.empty:
                 pd.DataFrame({"Message": ["Query executed successfully but returned no matching records."]}).to_excel(writer, index=False, sheet_name='Results')
            else:
                 results_df.to_excel(writer, index=False, sheet_name='Results')
        else:
            # Handle case where results_df is None (e.g., DB error or >10k rows before processing)
            pd.DataFrame({"Message": ["No results data to display due to an error or query limit."]}).to_excel(writer, index=False, sheet_name='Results')


        # Sheet 2: Query Info
        query_info_data = {
            'Item': ['User Question', 'Query Summary', 'Generated SQL Query'], # Added item
            'Details': [
                user_question if user_question else "N/A",
                query_summary if query_summary else "Summary not generated or available.",
                generated_sql if generated_sql else "SQL query not generated or available." # Added value
            ]
        }
        query_info_df = pd.DataFrame(query_info_data)
        query_info_df.to_excel(writer, index=False, sheet_name='Query Info')

        # Optional: Auto-adjust column widths for Query Info sheet
        # This will now adjust based on the potentially long SQL query as well
        worksheet = writer.sheets['Query Info']
        for idx, col in enumerate(query_info_df):  # loop through columns
            series = query_info_df[col]
            # Calculate max length needed for column, considering multi-line content split by newline
            max_len = 0
            if col == 'Details': # Special handling for the Details column which might contain long text/SQL
                 # Check max length of each line within each cell
                 for item in series.astype(str):
                      lines = item.split('\n')
                      for line in lines:
                           max_len = max(max_len, len(line))
                 max_len = max(max_len, len(str(series.name))) # Also consider header length
            else:
                 # Standard max length calculation for other columns
                 max_len = max((
                     series.astype(str).map(len).max(),  # len of largest item
                     len(str(series.name))               # len of column name/header
                 ))

            # Set width, add buffer, consider a maximum width if desired (e.g., 100)
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_len + 2, 100) # Add buffer, max width 100

        # Enable text wrapping for the 'Details' column (Column B)
        details_col_letter = 'B' # Assuming 'Details' is the second column (index 1 -> B)
        for row in worksheet.iter_rows(min_row=2, max_col=2, min_col=2): # Skip header row
            for cell in row:
                cell.alignment = openpyxl.styles.Alignment(wrap_text=True, vertical='top')


    processed_data = output.getvalue()
    return processed_data

# --- Initialize Session State ---
if 'generated_sql' not in st.session_state:
    st.session_state.generated_sql = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'summary_text' not in st.session_state:
    st.session_state.summary_text = None
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'llm_choice' not in st.session_state:
    st.session_state.llm_choice = "Chat GPT" # Default model


# --- Streamlit App Layout ---
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# --- Initialization of Session State ---
# =============================================================================
# Create a persistent conversation history list if it doesn't exist.
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Define some base states if they don't exist
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Chat GPT"

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# =============================================================================
# --- Streamlit Layout Configuration ---
# =============================================================================
st.set_page_config(page_title="Data Query Assistant", layout="wide")

# --- Sidebar ---
with st.sidebar:
    # Logo Placeholder
    try:
        st.image("logo.png", width=150)
    except Exception:
        st.warning("logo.png not found. Displaying text placeholder.")
        st.markdown("### Your Logo Here")

    # LLM Selection
    st.session_state.llm_choice = st.selectbox(
        "Select Model",
        ["Chat GPT", "Local Ollama"],
        index=0 if st.session_state.llm_choice == "Chat GPT" else 1,
        key="llm_select"
    )
    st.info(f"Using {st.session_state.llm_choice} for query generation.")
    if st.session_state.llm_choice == "Local Ollama":
        st.warning("Ensure the Ollama server is running and accessible.", icon="🔌")

# --- Main Content ---
st.title("📊 Data Query Assistant")
st.markdown(
    "Ask a question about the **Encounters** data in natural language. "
    "The AI will generate a SQL query, execute it against the database, and show you the results."
)

# --- Data Documentation (Collapsible) ---
with st.expander("View Encounter Data Schema"):
    st.markdown(
        """
        **Schema:** `testschema` | **Table:** `encounters`

        | Column Name           | Data Type     | Description                                   |
        |-----------------------|---------------|-----------------------------------------------|
        | Id                    | VARCHAR(50)   | Unique encounter identifier (Primary Key)   |
        | START                 | TIMESTAMP     | Encounter start date/time                     |
        | STOP                  | TIMESTAMP     | Encounter end date/time                       |
        | PATIENT               | VARCHAR(50)   | Unique patient identifier                     |
        | ORGANIZATION          | VARCHAR(100)  | Healthcare provider organization              |
        | PAYER                 | VARCHAR(100)  | Insurance company / Payer                     |
        | ENCOUNTERCLASS        | VARCHAR(50)   | Type of encounter (inpatient, outpatient, etc.)|
        | CODE                  | VARCHAR(50)   | Standardized encounter code                   |
        | DESCRIPTION           | TEXT          | Description of the encounter code             |
        | BASE_ENCOUNTER_COST   | NUMERIC(10,2) | Standard cost before adjustments              |
        | TOTAL_CLAIM_COST      | NUMERIC(10,2) | Total amount billed/claimed                   |
        | PAYER_COVERAGE        | NUMERIC(10,2) | Amount covered by the payer                   |
        | REASONCODE            | VARCHAR(50)   | Code for the reason for the encounter         |
        | REASONDESCRIPTION     | TEXT          | Description of the reason for the encounter   |

        *Ask questions based on these columns.*
        """
    )

# --- User Input Area ---
col1, col2 = st.columns([4, 1])  # Give more space to text area

with col1:
    user_question = st.text_area(
        "",
        height=100,
        key="user_question_input",
        placeholder="e.g., Show me the top 10 most expensive encounters based on total claim cost."
    )
with col2:
    st.write("")
    st.write("")  # Spacing
    get_results_button = st.button("🚀 Get Results", key="get_results_button", use_container_width=True)
    clear_button = st.button("🧹 Clear", key="clear_button", use_container_width=True)

# =============================================================================
# --- Processing a New Query ---
# =============================================================================
if get_results_button and user_question:
    # If the question is new or you wish to always append,
    # remove logic that clears previous results.
    st.session_state.last_question = user_question

    with st.spinner(f"Generating SQL query using {st.session_state.llm_choice}..."):
        if st.session_state.llm_choice == "Local Ollama":
            generated_sql = generate_sql_query_ollama(user_question)
        else:
            # Assuming the OpenAI API key is configured
            generated_sql = generate_sql_query_openai(user_question)

    if generated_sql:
        with st.spinner("Executing query against database..."):
            results_df = run_sql_query(generated_sql)
        with st.spinner("Summarizing query in plain English..."):
            # Only summarize if the required client is available
            summary_text = summarize_sql_query(generated_sql)

        # Create a timestamp for the current query
        query_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Package all the query details into a dictionary for history
        query_entry = {
            "timestamp": query_timestamp,
            "question": user_question,
            "generated_sql": generated_sql,
            "summary_text": summary_text,
            "results_df": results_df
        }
        # Append this entry to the conversation history list
        st.session_state.conversation_history.append(query_entry)

        st.success("✅ Query processed and saved to conversation history.")

# =============================================================================
# --- Displaying the Conversation History ---
# =============================================================================
if st.session_state.conversation_history:
    st.header("📝 Conversation History")
    # Iterate over the history; you can choose to reverse the list to show the most recent first.
    for entry in reversed(st.session_state.conversation_history):
        with st.expander(f"{entry['timestamp']} - {entry['question']}"):
            # Display query summary
            st.subheader("📄 Query Summary")
            if entry["summary_text"]:
                st.markdown(entry["summary_text"])
            else:
                st.info("Summary not available.")

            # Display query results
            st.subheader("📊 Query Results")
            if entry["results_df"] is not None:
                if not entry["results_df"].empty:
                    st.dataframe(entry["results_df"], use_container_width=True)
                    
                    # Prepare Excel data (including the context from the query)
                    excel_data = to_excel(
                        entry["results_df"],
                        entry["question"],
                        entry["summary_text"],
                        entry["generated_sql"]
                    )
                    # Create a filename using the timestamp (sanitize timestamp for filenames)
                    filename = f"query_results_{entry['timestamp'].replace(' ','_').replace(':','')}.xlsx"
                    st.download_button(
                        label="📥 Download Results as Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif "Status" in entry["results_df"].columns:
                    st.info(entry["results_df"]["Status"].iloc[0])
                else:
                    st.info("The query executed successfully but returned no matching records.")
            else:
                st.warning("No results available due to an error during query execution.")
            
            # Display generated SQL query
            st.subheader("🧠 Generated SQL Query")
            st.code(entry["generated_sql"], language='sql')
