import streamlit as st
import uuid
import requests

st.title("🛡️ Radicalization Detection Support Bot")

# Initialize session_id before displaying it
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Display the unique session ID so user can see and copy it
st.markdown(f"**Your session ID:** `{st.session_state['session_id']}`")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "conversation_started" not in st.session_state:
    st.session_state["conversation_started"] = False
if "lang" not in st.session_state:
    st.session_state["lang"] = "english"
if "region" not in st.session_state:
    st.session_state["region"] = ""

def send_message():
    user_text = st.session_state["user_input"]
    if user_text.strip():
        try:
            raw = requests.post("http://localhost:8000/chat", json={
                "session_id": st.session_state["session_id"],
                "message": user_text,
                "lang": st.session_state["lang"],
                "region": st.session_state["region"]
            })
            response = raw.json()

            if "reply" in response:
                st.session_state["messages"].append(("🧑 You", user_text))
                st.session_state["messages"].append(("🤖 Bot", response["reply"]))
                st.session_state["conversation_started"] = True
            elif "error" in response:
                st.error(f"Backend error: {response['error']}")
            else:
                st.error("Unexpected backend response.")
        except Exception as e:
            st.error(f"❌ Backend error or invalid JSON: {e}")

        st.session_state["user_input"] = ""

with st.form(key="message_form", clear_on_submit=True):
    lang = st.selectbox(
        "Choose language",
        ["english", "deutch"],
        key="lang",
        index=["english", "deutch"].index(st.session_state["lang"]) if st.session_state["lang"] in ["english", "deutch"] else 0,
        disabled=st.session_state["conversation_started"]  # disable language dropdown only
    )
    region = st.text_input(
        "Enter your region (e.g., berlin, nrw, bundesweit)",
        value=st.session_state["region"],
        key="region",
        disabled=False  # keep region always editable
    )
    user_input = st.text_area(
        "What would you like to talk about?",
        height=150,
        key="user_input"
    )
    submit_button = st.form_submit_button("Send", on_click=send_message)

if st.session_state["messages"]:
    st.markdown("### 💬 Conversation History")
    for speaker, msg in st.session_state["messages"]:
        st.markdown(f"**{speaker}:** {msg}")
