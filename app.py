import streamlit as st
import uuid
import requests

st.title("🛡️ Radicalization Detection Support Bot")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "conversation_started" not in st.session_state:
    st.session_state["conversation_started"] = False

if "lang" not in st.session_state:
    st.session_state["lang"] = "english"

if "region" not in st.session_state:
    st.session_state["region"] = ""

# Show session ID
st.markdown(f"**Your session ID:** `{st.session_state['session_id']}`")

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

# Choose language (outside form so it affects labels immediately)
current_lang = st.selectbox(
    "Choose language / Sprache wählen",
    ["english", "deutch"],
    index=["english", "deutch"].index(st.session_state["lang"]) if st.session_state["lang"] in ["english", "deutch"] else 0,
    key="lang",
    disabled=st.session_state["conversation_started"]
)

# Set prompt labels according to selected language
if current_lang == "deutch":
    region_label = "📍 Was ist deine Region? (z.B. berlin, nrw, bremen):"
    concern_label = (
        "💬 Dieser Chatbot soll Ihnen dabei helfen, Ihre Bedenken hinsichtlich einer Person zu untersuchen, "
        "die erste Anzeichen einer Radikalisierung zeigt, insbesondere in Richtung Rechtsextremismus. "
        "Er soll einfühlsam, anonym und vorurteilsfrei sein. (Tippe 'exit' zum Beenden):"
    )
else:
    region_label = "📍 What's your region? (e.g. berlin, nrw, bremen):"
    concern_label = (
        "💬 This chat-bot is here to assist you in investigating any worries you may have regarding someone "
        "exhibiting early indications of radicalization, particularly in the direction of right-wing extremism. "
        "It is intended to be sympathetic, anonymous, and nonjudgmental. (type 'exit' to quit):"
    )

# Input form for region and concern
with st.form(key="message_form", clear_on_submit=True):
    region = st.text_input(
        region_label,
        value=st.session_state["region"],
        key="region",
        disabled=False
    )

    user_input = st.text_area(
        concern_label,
        height=150,
        key="user_input"
    )

    submit_button = st.form_submit_button("Send", on_click=send_message)

# Display chat history
if st.session_state["messages"]:
    st.markdown("### 💬 Conversation History")
    for speaker, msg in st.session_state["messages"]:
        st.markdown(f"**{speaker}:** {msg}")
