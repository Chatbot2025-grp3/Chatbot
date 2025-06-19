import streamlit as st
import uuid
import requests
import os

# Localized region options with placeholder
REGIONS = {
    "en": [
        "--select your region--",
        "Baden-Württemberg", "Bayern", "Berlin", "Bremen", "Bundesweit",
        "Hamburg", "Jena", "Mecklenburg-Vorpommern", "Niedersachen",
        "Nordrhein Westfalen", "Potsdam", "Rheinland Pfalz", "Sachsen-Anhalt",
        "St. Wendel", "Thüringen", "bundesweit"
    ],
    "de": [
        "--bitte Region wählen--",
        "Baden-Württemberg", "Bayern", "Berlin", "Bremen", "Bundesweit",
        "Hamburg", "Jena", "Mecklenburg-Vorpommern", "Niedersachen",
        "Nordrhein Westfalen", "Potsdam", "Rheinland Pfalz", "Sachsen-Anhalt",
        "St. Wendel", "Thüringen", "bundesweit"
    ]
}

st.set_page_config(page_title="Radicalization Concern Bot", layout="centered")
st.title("🛡️ Radicalization Detection Support Bot")

# Session initialization
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "conversation_started" not in st.session_state:
    st.session_state["conversation_started"] = False
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "region" not in st.session_state:
    st.session_state["region"] = ""

if (
    not st.session_state["conversation_started"]
    and st.session_state["lang"] is not None
    and st.session_state["region"] not in ["", REGIONS[st.session_state["lang"]][0]]
):
    st.session_state["conversation_started"] = True


# Language and region setup
if not st.session_state["conversation_started"]:
    st.markdown(f"**Your Session ID:** `{st.session_state['session_id']}`")
    st.subheader("🌐 Start Your Conversation")

    language = st.selectbox(
        "Choose Language / Sprache wählen:",
        ["English", "Deutsch"],
        index=0,
        key="lang_select"
    )
    lang_code = "en" if language == "English" else "de"

    region = st.selectbox(
        "📍 Choose your Region:" if lang_code == "en" else "📍 Wähle deine Region:",
        REGIONS[lang_code],
        index=0,
        key="region_select"
    )

    if region == REGIONS[lang_code][0]:
        st.warning("Please select a valid region before proceeding." if lang_code == "en" else "Bitte wähle eine gültige Region aus.")
    else:
        if st.button("✅ Confirm"):
            st.session_state["lang"] = lang_code
            st.session_state["region"] = region
            st.session_state["conversation_started"] = True
            st.rerun()

else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Your Session ID:** `{st.session_state['session_id']}`")
    with col2:
        new_lang = st.selectbox("🌐", ["English", "Deutsch"], index=0 if st.session_state["lang"] == "en" else 1, label_visibility="collapsed")
        new_lang_code = "en" if new_lang == "English" else "de"
        if new_lang_code != st.session_state["lang"]:
            st.session_state["lang"] = new_lang_code
            st.session_state["messages"].append(("bot", f"🌐 Language has been changed to {'English' if new_lang_code == 'en' else 'Deutsch'}."))

    # 🔧 Get API URL from environment
    API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost")
    API_PORT = os.environ.get("API_PORT", "8001")

    if "localhost" in API_BASE_URL or "127.0.0.1" in API_BASE_URL:
        API_URL = f"{API_BASE_URL}:8000/chat"
    else:
        API_URL = f"{API_BASE_URL}:{API_PORT}/chat"


    def send_message():
        user_text = st.session_state["user_input"].strip()
        if user_text:
            try:
                raw = requests.post(API_URL, json={
                    "session_id": st.session_state["session_id"],
                    "message": user_text,
                    "lang": st.session_state["lang"],
                    "region": st.session_state["region"]
                })
                response = raw.json()
                if "reply" in response:
                    st.session_state["messages"].append(("user", user_text))
                    st.session_state["messages"].append(("bot", response["reply"]))
            except Exception as e:
                st.error(f"Backend error: {e}")
            st.session_state["user_input"] = ""

    st.markdown("""
    💬 This chat-bot is here to assist you in investigating any worries you may have regarding someone exhibiting early indications of radicalization, particularly in the direction of right-wing extremism. It is intended to be sympathetic, anonymous, and nonjudgmental. (type 'exit' to quit):
    """ if st.session_state["lang"] == "en" else """
    💬 Dieser Chatbot soll Ihnen dabei helfen, Ihre Bedenken hinsichtlich einer Person zu untersuchen, die erste Anzeichen einer Radikalisierung zeigt, insbesondere in Richtung Rechtsextremismus. Er soll einfühlsam, anonym und vorurteilsfrei sein. (Tippe 'exit' zum Beenden):
    """)

    for sender, msg in st.session_state["messages"]:
        if sender == "user":
            st.markdown(f"<div style='text-align:right; background-color:#dcf8c6; padding:10px; border-radius:10px; color:black;'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; background-color:#f1f0f0; padding:10px; border-radius:10px; color:black;'>{msg}</div>", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("", height=80, key="user_input")
        submit = st.form_submit_button("Send", on_click=send_message)
