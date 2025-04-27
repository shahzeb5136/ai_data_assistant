# app.py (Corrected and with Viz page link)

import streamlit as st
import datetime

# --- Page Configuration (Set ONCE for the entire app) ---
st.set_page_config(
    page_title="AI Data Assistant Hub",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:your_support_email@example.com', # Optional
        'Report a bug': "mailto:your_bug_report_email@example.com", # Optional
        'About': """
         ## AI Data Assistant Hub
         This application provides AI-powered tools for interacting with data.
         Developed using Streamlit, OpenAI, and Ollama.
         """ # Optional
    }
)

# --- Sidebar Content (Common elements across pages) ---
with st.sidebar:
    # --- Logo ---
    try:
        st.image("logo.png", width=150)
    except Exception:
        st.warning("logo.png not found. Displaying text placeholder.", icon="🖼️")
        st.markdown("### Your Logo Here")

    st.markdown("---") # Divider
    st.subheader("Navigation")
    st.markdown("Select a tool from the pages listed above.")
    st.markdown("---") # Divider

    # --- Display Current Date (Reliable way) ---
    # This line correctly shows the current date
    st.info(f"Today is: {datetime.date.today().strftime('%Y-%m-%d')}")


# --- Main Page Content ---

st.title("Welcome to the AI Data Assistant Hub!")

st.markdown(
    """
    This hub provides intelligent tools designed to help you interact with and understand your data more effectively.
    Whether you need to query databases using natural language, extract information from documents, or visualize results,
    the AI assistants are here to help.

    **Please use the navigation sidebar on the left to choose your desired tool.**
    """
)

st.divider() # Visual separator

st.header("Explore The Tools:")

# Use columns for a cleaner layout for the main tools
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📊 SQL Query Assistant")
    # You can find icons online, e.g., flaticon.com, or remove the st.image line if you prefer text only
    st.image("https://cdn-icons-png.flaticon.com/512/9101/9101491.png", width=80) # Example icon
    st.markdown(
        """
        * **Ask questions** about your database tables using everyday language.
        * The AI **generates the correct SQL query** for you.
        * **View results** directly in the app.
        * Get a **plain-English summary** of what the query does.
        * **Download** your data easily as an Excel file (ready for the Visualizer!).
        """
    )
    st.page_link("pages/sql.py", label="Go to SQL Assistant", icon="➡️")

with col2:
    st.subheader("📄 PDF Query Assistant (WIP)")
    # Example icon
    st.image("https://cdn-icons-png.flaticon.com/512/337/337946.png", width=80)
    st.markdown(
        """
        * **Upload** your PDF documents.
        * **Ask specific questions** about the content within the PDF.
        * The AI reads the document and **provides answers based *only* on its text**.
        * Choose between **OpenAI's powerful models or a local Ollama instance**.
        * Ideal for quickly finding information in reports, papers, or manuals.
        """
    )
    st.page_link("pages/pdf.py", label="Go to PDF Assistant", icon="➡️")

st.divider() # Visual separator

# --- Added Visualization Section ---
st.subheader("📈 Data Visualizer (WIP)")
# Example chart icon
st.image("https://cdn-icons-png.flaticon.com/512/1379/1379888.png", width=80)
st.markdown(
    """
    * **Upload** your data file (e.g., the Excel file downloaded from the SQL Assistant).
    * Select columns and chart types to **create insightful visualizations**.
    * Explore your data visually with bar charts, line charts, scatter plots, and more.
    * Currently requires manual **download from SQL Assistant and upload here**.
    """
)
# Ensure your visualization page is named viz.py and located in the pages folder
st.page_link("pages/viz.py", label="Go to Data Visualizer", icon="➡️")


st.divider()

# --- Additional Information / Footer ---
st.markdown(
    """
    **Getting Started:**
    1.  Select the assistant you need from the sidebar navigation.
    2.  Follow the instructions provided on that specific tool's page.
    3.  Ensure any necessary connections (like database access or a running Ollama server) are active.

    *Happy Analyzing!*
    """
)

# Note on Ollama - Useful reminder if applicable
st.warning(
    """
    **Note on Local Models:** If you plan to use the 'Local Ollama' option in either tool,
    please ensure your [Ollama](https://ollama.com/) server is running locally and is accessible
    (typically at `http://localhost:11434`).
    """,
    icon="🔌"
)