import streamlit as st
import pandas as pd
from openai import OpenAI
import traceback
import io
import seaborn as sns
import matplotlib.pyplot as plt
import re
from typing import Optional, Tuple, Dict, Any

# --- Configuration ---
st.set_page_config(page_title="📊 Data Visualizer", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid") # Apply a nice default theme

# --- Constants ---
RESULTS_SHEET = "Results"
QUERY_INFO_SHEET = "Query Info"
VISUALIZATION_MODEL = "gpt-4o" # Or "gpt-3.5-turbo", etc.

# --- Session State Initialization ---
if 'viz_code' not in st.session_state:
    st.session_state.viz_code = None
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'user_request_active' not in st.session_state:
    st.session_state.user_request_active = False
if 'ai_suggestion_active' not in st.session_state:
    st.session_state.ai_suggestion_active = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# --- OpenAI Client Initialization ---
# User explicitly asked to leave API key handling as is for now.
# In a real application, use st.secrets or environment variables.
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
client: Optional[OpenAI] = None
api_key_error = None

try:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
         raise ValueError("API Key is missing or still set to the placeholder.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Perform a simple test call to verify the key (optional but recommended)
    # client.models.list() # Uncomment to test key on startup
    st.sidebar.success("OpenAI client initialized.", icon="✅")
except (KeyError, ValueError) as e:
    api_key_error = f"OpenAI API Key Error: {e}. Please ensure it's correctly set."
    st.sidebar.error(api_key_error)
except Exception as e: # Catch other potential OpenAI client errors
    api_key_error = f"Error initializing OpenAI client: {e}"
    st.sidebar.error(api_key_error)


# --- Prompts ---
SYSTEM_PROMPT_AI_SUGGESTION = """
You are an expert Python data visualization assistant specializing in the Seaborn library.
You will receive a description of a dataset derived from a user's question and an SQL query summary. You will also receive the column names and the first few rows of the data itself.
Your task is to generate *only* the Python code (using Seaborn) that creates the *single most appropriate and insightful* visualization for the described data and context.

**Instructions:**
1. Analyze the data description, column names, data types (implied from the head()), and the context (user question/summary) to determine the best plot type (e.g., bar plot, scatter plot, histogram, line plot, box plot, count plot etc.).
2. Assume the data is already loaded into a pandas DataFrame named `df`.
3. Assume `seaborn as sns` and `matplotlib.pyplot as plt` have already been imported.
4. Generate *only* the Python code for creating the plot.
5. Do **NOT** include any explanations, comments within the code (unless essential for clarity like axis labels), or markdown formatting (like ```python ... ```).
6. Focus on clarity, relevance, and readability of the plot.
7. Set appropriate labels for axes using `plt.xlabel()`, `plt.ylabel()`, and a title using `plt.title()`. Use informative labels based on column names/context.
8. If dealing with categorical data on the x-axis with potentially many labels, rotate them: `plt.xticks(rotation=45, ha='right')`.
9. Use `plt.tight_layout()` to prevent labels from overlapping.
10. The final line of your code should be the Seaborn plotting command itself (e.g., `sns.barplot(...)`). Do **NOT** include `plt.show()` or `plt.figure()`. Streamlit handles the figure display and creation.

**Example Input Context:**
- User Question: what are the top 20 ids by cost in 2019. and include the total cost and payer coverage.
- Query Summary: This query retrieves the top 20 encounters from the year 2019, sorted by the highest total claim cost. It includes the unique encounter ID, the total amount billed for the encounter, and the portion covered by the payer.
- Columns: ['Id', 'TOTAL_CLAIM_COST', 'PAYER_COVERAGE']
- Data Head:
   Id | TOTAL_CLAIM_COST | PAYER_COVERAGE
 ---|-----------------|----------------
  A1 |        15000.50 |        12000.00
  B2 |        14500.75 |        11500.25
  C3 |        14000.00 |        10000.00

**Example Output (Code Only):**
sns.barplot(data=df.head(10), x='Id', y='TOTAL_CLAIM_COST', palette='viridis') # Limit display if too many IDs
plt.xlabel('Encounter ID (Top 10)')
plt.ylabel('Total Claim Cost ($)')
plt.title('Top 10 Encounters by Total Claim Cost')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
"""

SYSTEM_PROMPT_USER_REQUEST = """
You are an expert Python data visualization assistant specializing in the Seaborn library.
You will receive a user's specific request for a type of plot, a description of the dataset (derived from an initial question and query summary), the column names, and the first few rows of the data.
Your task is to generate *only* the Python code (using Seaborn) that fulfills the user's plot request using the provided data.

**Instructions:**
1. Prioritize the user's specific plot request (e.g., "create a scatter plot", "show a histogram").
2. Use the provided column names and data sample to correctly map data to plot axes and elements according to the request.
3. Assume the data is already loaded into a pandas DataFrame named `df`.
4. Assume `seaborn as sns` and `matplotlib.pyplot as plt` have already been imported.
5. Generate *only* the Python code for creating the plot.
6. Do **NOT** include any explanations, comments within the code (unless essential for clarity like axis labels), or markdown formatting (like ```python ... ```).
7. Set appropriate labels for axes using `plt.xlabel()`, `plt.ylabel()`, and a title using `plt.title()`. Make the title reflect the user's request and the data shown.
8. If dealing with categorical data on the x-axis with potentially many labels, rotate them: `plt.xticks(rotation=45, ha='right')`.
9. Use `plt.tight_layout()` to prevent labels from overlapping.
10. The final line of your code should be the Seaborn plotting command itself (e.g., `sns.scatterplot(...)`). Do **NOT** include `plt.show()` or `plt.figure()`. Streamlit handles the figure display and creation.
11. If the user's request is ambiguous or cannot be fulfilled with the provided columns (e.g., requesting a plot using a non-existent column), generate a simple, valid plot (like `sns.histplot(data=df, x=df.columns[0])` if possible) and add a comment within the code `# User request might be unclear or require different data.`.

**Example Input Context:**
- User Plot Request: Make a scatter plot showing the relationship between total cost and payer coverage.
- User Question: what are the top 20 ids by cost in 2019. and include the total cost and payer coverage.
- Query Summary: This query retrieves the top 20 encounters from the year 2019, sorted by the highest total claim cost...
- Columns: ['Id', 'TOTAL_CLAIM_COST', 'PAYER_COVERAGE']
- Data Head:
   Id | TOTAL_CLAIM_COST | PAYER_COVERAGE
 ---|-----------------|----------------
  A1 |        15000.50 |        12000.00
  B2 |        14500.75 |        11500.25
  C3 |        14000.00 |        10000.00

**Example Output (Code Only):**
sns.scatterplot(data=df, x='TOTAL_CLAIM_COST', y='PAYER_COVERAGE')
plt.xlabel('Total Claim Cost ($)')
plt.ylabel('Payer Coverage ($)')
plt.title('Relationship between Total Claim Cost and Payer Coverage')
plt.tight_layout()
"""


# --- Helper Functions ---

@st.cache_data # Cache the data reading
def read_excel_sheets(uploaded_file_content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """Reads 'Results' and 'Query Info' sheets from uploaded Excel file content."""
    error_msg = None
    try:
        excel_data = pd.ExcelFile(uploaded_file_content) # Read from bytes
        if RESULTS_SHEET not in excel_data.sheet_names:
            error_msg = f"Error: '{RESULTS_SHEET}' sheet not found in '{filename}'."
            st.error(error_msg)
            return None, None, error_msg
        if QUERY_INFO_SHEET not in excel_data.sheet_names:
            error_msg = f"Error: '{QUERY_INFO_SHEET}' sheet not found in '{filename}'."
            st.error(error_msg)
            return None, None, error_msg

        results_df = pd.read_excel(excel_data, sheet_name=RESULTS_SHEET)
        query_info_df = pd.read_excel(excel_data, sheet_name=QUERY_INFO_SHEET)

        # --- Data Validation ---
        if not all(item in query_info_df.columns for item in ['Item', 'Details']):
            error_msg = "Error: 'Query Info' sheet columns must include 'Item' and 'Details'."
            st.error(error_msg)
            return None, None, error_msg

        if results_df.empty:
            # Check if it's intentionally empty with a message (like "no records found")
             if "Message" not in results_df.columns:
                 st.warning(f"⚠️ The '{RESULTS_SHEET}' sheet in '{filename}' is empty. No data to visualize.")
             # If it has only a "Message" column, it's likely a status message - handled later.

        elif "Message" in results_df.columns and len(results_df) == 1:
            message = results_df["Message"].iloc[0]
            st.info(f"ℹ️ Status from '{RESULTS_SHEET}': {message}")
            # Return the DFs anyway, but visualization will be blocked later

        # Attempt basic type inference/conversion for numeric columns if possible
        # This helps the AI and plotting functions
        for col in results_df.select_dtypes(include=['object']).columns:
            try:
                results_df[col] = pd.to_numeric(results_df[col])
                st.info(f"ℹ️ Converted column '{col}' to numeric.", icon="🔄")
            except (ValueError, TypeError):
                pass # Keep as object if conversion fails

        return results_df, query_info_df, None

    except Exception as e:
        error_msg = f"Error reading Excel file '{filename}': {e}"
        st.error(error_msg)
        st.error(traceback.format_exc()) # Log detailed error
        return None, None, error_msg

def get_visualization_code(
    client_instance: OpenAI,
    user_question: str,
    query_summary: str,
    results_df: pd.DataFrame,
    user_plot_request: Optional[str] = None
) -> Optional[str]:
    
    """Sends context to OpenAI API and returns the generated Python code string."""
    global api_key_error # Allow modification of the global variable

    if api_key_error: # Check if client failed to initialize
        st.error(f"Cannot generate visualization: {api_key_error}")
        return None

    if client_instance is None:
         st.error("Cannot generate visualization: OpenAI client is not available.")
         return None

    if results_df is None or results_df.empty:
        st.warning("Cannot generate visualization code: No valid 'Results' data available.")
        return None
    if "Message" in results_df.columns and len(results_df) == 1:
        st.warning(f"Cannot generate visualization code: Results contain a status message: {results_df['Message'].iloc[0]}")
        return None

    # Prepare context for the LLM
    column_names = results_df.columns.tolist()
    data_head_markdown = results_df.head().to_markdown(index=False)

    # Choose prompt and add user request if provided
    if user_plot_request:
        system_prompt = SYSTEM_PROMPT_USER_REQUEST
        user_content = f"""
        **User's Specific Plot Request:**
        {user_plot_request}

        --- Data Context ---
        **User's Original Question:**
        {user_question}
        **Query Summary:**
        {query_summary}
        **Data Columns:**
        {column_names}
        **Data Sample (first 5 rows):**
        ```markdown
        {data_head_markdown}
        ```
        Based on the user's request and the data context, provide the Python code using Seaborn to create the specified visualization.
        Remember to follow all instructions from the system prompt (assume df, sns, plt exist; only code output; labels/title; no plt.show()/plt.figure()).
        """
        spinner_message = f"🧠 Asking AI to generate code for: '{user_plot_request}'..."
    else:
        system_prompt = SYSTEM_PROMPT_AI_SUGGESTION
        user_content = f"""
        Here is the context for the data visualization:
        **User's Original Question:**
        {user_question}
        **Query Summary:**
        {query_summary}
        **Data Columns:**
        {column_names}
        **Data Sample (first 5 rows):**
        ```markdown
        {data_head_markdown}
        ```
        Based on this information, provide the Python code using Seaborn to create the single best visualization for this data.
        Remember to follow all instructions from the system prompt (assume df, sns, plt exist; only code output; labels/title; no plt.show()/plt.figure()).
        """
        spinner_message = "🧠 Asking AI for the best visualization suggestion..."

    try:
        with st.spinner(spinner_message):
            response = client_instance.chat.completions.create(
                model=VISUALIZATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2, # Lower temperature for more deterministic code
                stream=False
            )
            generated_code = response.choices[0].message.content.strip()

            # Clean potential markdown backticks and python keyword
            generated_code = re.sub(r"^```python", "", generated_code, flags=re.IGNORECASE).strip()
            generated_code = re.sub(r"```$", "", generated_code).strip()

            # Basic validation
            if not generated_code or not ("sns." in generated_code or "plt." in generated_code):
                st.error("Error: The AI did not return valid-looking Python code.")
                st.code(generated_code or "No code returned.", language='text')
                return None

            # Check for potential refusal or inability message from the LLM
            if "cannot create a plot" in generated_code.lower() or "unable to generate" in generated_code.lower():
                 st.warning(f"AI indicated it couldn't generate the plot. Reason:\n```\n{generated_code}\n```")
                 return None


            return generated_code

    except Exception as e:
        st.error(f"Error communicating with OpenAI API: {e}")
        st.error(traceback.format_exc())
        st.session_state.error_message = f"API Error: {e}" # Store error message
        return None

def execute_and_plot(code_string: str, df: pd.DataFrame, plot_placeholder: st.empty, code_placeholder: st.empty):
    """Executes the generated Python code and displays the plot."""
    if not code_string:
        st.warning("No code provided to execute.")
        return False # Indicate failure

    # Display the code first
    code_placeholder.code(code_string, language='python')

    fig = None # Initialize fig to None
    try:
        # Prepare execution scope
        # Pass only necessary libraries and the dataframe `df`
        execution_globals = {'sns': sns, 'plt': plt, 'pd': pd, 'df': df}
        # Using a restricted scope is slightly safer than passing globals()
        # but exec() still carries risks.
        execution_locals = {}

        # Clear any previous plot artefacts before execution
        plt.close('all') # Close all previous figures
        fig = plt.figure() # Create a new figure context for the code to use

        # --- Execute the generated code string ---
        # !!! SECURITY WARNING: exec() can run arbitrary code. !!!
        # !!! Only use this in trusted environments.           !!!
        st.warning("⚠️ **Security Note:** Executing AI-generated code can be risky. Ensure you trust the source and have reviewed the code.", icon="❗")
        with st.spinner("Executing generated code..."):
            exec(code_string, execution_globals, execution_locals)

        # --- Plotting ---
        # Check if the current figure (captured or default) has axes (meaning plotted)
        if fig and fig.get_axes():
            plot_placeholder.pyplot(fig)
            st.success("✅ Visualization generated successfully!")
            plt.close(fig) # Close the figure after displaying to free memory
            return True # Indicate success
        else:
            # Check if any figures were created but maybe not captured
            if plt.get_fignums(): # Check if any matplotlib figures exist
                 plot_placeholder.pyplot(plt.gcf()) # Try getting the current figure again
                 st.success("✅ Visualization generated successfully (alternate capture).")
                 plt.close(plt.gcf()) # Close the figure
                 return True # Indicate success
            else:
                 st.warning("Code executed, but no plottable output was detected.")
                 code_placeholder.code(code_string, language='python') # Show code again
                 return False # Indicate failure


    except Exception as e:
        st.error(f"❌ Error executing the generated visualization code:")
        st.error(traceback.format_exc())
        # Show the problematic code again for context
        st.error("Problematic Code:")
        code_placeholder.code(code_string, language='python')
        st.session_state.error_message = f"Execution Error: {e}" # Store error
        if fig:
            plt.close(fig) # Ensure figure is closed on error
        return False # Indicate failure
    finally:
        # Ensure all figures potentially created are closed if not handled above
        plt.close('all')


# --- Streamlit App Layout ---
st.title("📊 Excel Data Visualizer Pro")
st.markdown(
"""
Upload an Excel file containing '**Results**' (data) and '**Query Info**' (context) sheets retreived from the SQL Querying assistant.
The app can either **suggest** a visualization based on the data and context, or you can
**request a specific plot** type (e.g., 'bar chart of cost per ID', 'scatter plot of X vs Y').
"""
)

# --- Sidebar ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader(
    "1. Choose an Excel file",
    type=["xlsx"],
    accept_multiple_files=False,
    key="excel_uploader"
)

if st.sidebar.button("Clear Cache & Reset App", key="clear_cache"):
    st.cache_data.clear()
    st.session_state.clear() # Clear all session state
    st.rerun()

# --- Main Logic ---
if uploaded_file is not None:
    # Check if this is a new file upload; if so, reset state
    if uploaded_file.name != st.session_state.get('last_uploaded_filename'):
        st.session_state.clear() # Clear state for new file
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.success(f"✅ New file '{uploaded_file.name}' uploaded. Resetting state.")


    # Read data from Excel using the content
    results_df, query_info_df, read_error = read_excel_sheets(uploaded_file.getvalue(), uploaded_file.name)

    if read_error:
        st.error(f"File Processing Error: {read_error}")
        # Stop further processing if file reading failed critically
        st.stop()

    if results_df is not None and query_info_df is not None:
        st.markdown("---")
        st.subheader("📂 Data Overview")

        # Use columns for better layout
        col_info, col_data = st.columns([1, 2])

        with col_info:
            st.markdown("#### 📄 Query Information")
            try:
                query_info_dict = query_info_df.set_index('Item')['Details'].to_dict()
                user_q = query_info_dict.get('User Question', 'N/A')
                q_summary = query_info_dict.get('Query Summary', 'N/A')
                sql_query = query_info_dict.get('Generated SQL Query', 'N/A')

                st.markdown(f"**User Question:**")
                st.markdown(f"> _{user_q}_")
                st.markdown(f"**Query Summary:**")
                st.markdown(f"> _{q_summary}_")
                with st.expander("Generated SQL Query (if available)"):
                    st.code(sql_query if sql_query != 'N/A' else "# No SQL Query provided", language='sql')
            except KeyError:
                 st.error("Error processing 'Query Info'. Make sure it has 'Item' and 'Details' columns.")
                 query_info_dict = {} # Prevent errors later
                 user_q, q_summary = "Error", "Error"


        with col_data:
            st.markdown(f"#### 📈 Results Data Preview (`{RESULTS_SHEET}` sheet)")
            # Check again for message row or empty dataframe
            is_message_row = "Message" in results_df.columns and len(results_df) == 1
            is_empty = results_df.empty

            if is_message_row:
                st.info(f"Results Status: {results_df['Message'].iloc[0]}")
            elif is_empty:
                st.warning("Results data is empty.")
            else:
                st.dataframe(results_df.head(), use_container_width=True)
                st.caption(f"Displaying first 5 rows of {len(results_df)} total rows.")

        # --- Visualization Section ---
        st.markdown("---")
        st.subheader("🎨 Visualization Generation")

        # Only proceed if there's data to visualize
        can_visualize = results_df is not None and not results_df.empty and not is_message_row

        if not can_visualize:
            st.info("No data available in the 'Results' sheet to visualize.")
        elif api_key_error: # Check if OpenAI client is working
             st.error(f"Cannot proceed with visualization due to API key/client issue: {api_key_error}")
        else:
            # --- User Choice: AI Suggestion vs. User Request ---
            viz_mode = st.radio(
                "Choose visualization mode:",
                ("Let AI Suggest the Best Plot", "Request a Specific Plot"),
                key="viz_mode_radio",
                horizontal=True,
            )

            user_plot_request = None
            if viz_mode == "Request a Specific Plot":
                user_plot_request = st.text_input(
                    "Describe the plot you want (e.g., 'scatter plot of cost vs coverage', 'histogram of total cost')",
                    key="user_plot_request_input",
                    placeholder="E.g., 'bar chart of count per category'"
                )
                if st.button("Generate My Plot Request", key="generate_user_request_btn", type="primary"):
                    if user_plot_request:
                        st.session_state.viz_code = get_visualization_code(
                            client, user_q, q_summary, results_df, user_plot_request
                        )
                        st.session_state.user_request_active = True # Flag that user request was made
                        st.session_state.ai_suggestion_active = False # Deactivate other mode flag
                        st.session_state.error_message = None # Reset error on new attempt
                    else:
                        st.warning("Please describe the plot you want.")

            else: # AI Suggestion Mode
                if st.button("Generate AI Suggested Plot", key="generate_ai_suggest_btn", type="primary"):
                    st.session_state.viz_code = get_visualization_code(
                        client, user_q, q_summary, results_df
                    )
                    st.session_state.ai_suggestion_active = True # Flag that AI suggestion was made
                    st.session_state.user_request_active = False # Deactivate other mode flag
                    st.session_state.error_message = None # Reset error on new attempt

            # --- Display Code and Plot ---
            # Check if code exists from either button press in the current run or previous state
            if st.session_state.get('viz_code'):
                display_code = st.session_state.viz_code
                mode_text = "User-Requested" if st.session_state.get('user_request_active') else "AI-Suggested"

                st.markdown(f"#### **{mode_text} Visualization**")
                st.markdown("**🤖 Generated Python Code:**")
                code_placeholder = st.empty() # Placeholder for code
                st.markdown("**🖼️ Generated Plot:**")
                plot_placeholder = st.empty() # Placeholder for plot

                # Execute and plot
                execute_and_plot(display_code, results_df, plot_placeholder, code_placeholder)

                # Show errors if they occurred during generation or execution
                if st.session_state.get('error_message'):
                    st.error(f"An error occurred: {st.session_state.error_message}")


            # If a button was clicked but code generation failed (viz_code is None)
            elif st.session_state.get('ai_suggestion_active') or st.session_state.get('user_request_active'):
                 if not st.session_state.get('error_message'): # Avoid double message if API error shown elsewhere
                    st.error("Failed to generate visualization code. The AI might have been unable to process the request or data.")


elif st.session_state.get('last_uploaded_filename'):
     # If no file is currently uploaded, but one was previously, show a message
     st.info("Please re-upload an Excel file to continue.")
else:
    st.info("Upload an Excel file using the sidebar to get started.")