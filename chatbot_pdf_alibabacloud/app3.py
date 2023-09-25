import sqlite3
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# # Session State Handling
# class SessionState(object):
#     def __init__(self, **kwargs):
#         for key, val in kwargs.items():
#             setattr(self, key, val)


# def get_state(**kwargs):
#     session_id = st.report_thread.get_report_ctx().session_id
#     session = st.server.server.Server.get_current()._get_session_info(session_id).session
#     if not hasattr(session, "_session_state"):
#         session._session_state = SessionState(**kwargs)
#     return session._session_state


# Error Handling for DB Operations
def run_query(query, params=()):
    conn = sqlite3.connect("conversation_history.db")
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        conn.commit()
        return cursor
    except sqlite3.Error as e:
        st.warning(f"Database error: {e}")
        return None
    finally:
        if conn:
            # Do not close the connection here
            pass


# Functions related to database operations
def create_database():
    run_query('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY,
                 username TEXT UNIQUE,
                 password TEXT
              )''')
    run_query('''CREATE TABLE IF NOT EXISTS history (
                 id INTEGER PRIMARY KEY,
                 user_id INTEGER,
                 role TEXT,
                 message TEXT,
                 FOREIGN KEY(user_id) REFERENCES users(id)
              )''')


def insert_user(username, password):
    run_query("INSERT INTO users (username, password) VALUES (?, ?)",
              (username, password))


def get_user(username):
    cursor = run_query("SELECT * FROM users WHERE username=?", (username,))
    if cursor:
        result = cursor.fetchone()
        cursor.close()  # Close the cursor here
        return result
    return None


def insert_message(user_id, role, message):
    run_query("INSERT INTO history (user_id, role, message) VALUES (?, ?, ?)",
              (user_id, role, message))


def retrieve_conversation(user_id):
    return run_query("SELECT * FROM history WHERE user_id=?", (user_id,)).fetchall()


def main(user_id):
    load_dotenv()

    st.set_page_config(page_title="Chatbot", layout="wide")
    st.title("💬 Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI LLM")

    OPENAI_API_KEY = "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"

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Create the SQLite database and table
    create_database()

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("You:", key="user_input")

    if st.button("Send") and user_question:
        if pdf:
            docs = knowledge_base.similarity_search(user_question)

            with get_openai_callback() as cb:
                # Perform question-answering
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(
                    input_documents=docs, question=user_question)

                # Display the response on the Streamlit app
                st.write(f"Assistant: {response}")

            insert_message(user_id, "User", user_question)
            insert_message(user_id, "Assistant", response)
        else:
            st.warning("Please upload a PDF to get a response.")

    # Add button to view conversation history
    if st.button("View Conversation History"):
        st.experimental_set_query_params(history_page=True)
        st.experimental_rerun()

    if st.button("Logout"):
        st.session_state.logged_in = False  # Reset the logged_in state
        st.session_state.user_id = None  # Reset the user_id state
        st.experimental_rerun()  # Rerun the script to reflect the state change


def login_page():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = get_user(username)
        if user and user[2] == password:
            st.session_state.logged_in = True
            st.session_state.user_id = user[0]
            st.success("Logged in successfully!")
            st.experimental_rerun()  # Rerun the script to reflect the state change
        else:
            st.warning("Incorrect username or password")


def register_page():
    st.subheader("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password == confirm_password:
            try:
                insert_user(username, password)
                st.success("Registration successful!")
            except sqlite3.IntegrityError:
                st.warning("Username already taken!")
        else:
            st.warning("Passwords do not match!")


if __name__ == '__main__':
    create_database()

    if not hasattr(st.session_state, 'logged_in'):
        st.session_state.logged_in = False
    if not hasattr(st.session_state, 'user_id'):
        st.session_state.user_id = None
    if st.session_state.logged_in:
        if "history_page" in st.experimental_get_query_params():
            # Show the Conversation History page
            st.title("💬 Conversation History")
            st.subheader("View and search your conversation history here")
            stored_conversation = retrieve_conversation(
                st.session_state.user_id)
            for idx, row in enumerate(stored_conversation):
                if row[2] == "User":
                    st.write(f"You_{idx}:", row[3])
                else:
                    st.write(f"Assistant_{idx}:", row[3])
        else:
            main(st.session_state.user_id)
    else:
        menu = ["Login", "Register"]
        choice = st.sidebar.selectbox("Menu", menu)
        if choice == "Login":
            login_page()
        elif choice == "Register":
            register_page()
