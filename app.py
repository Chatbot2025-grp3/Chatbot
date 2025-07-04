import streamlit as st
import uuid
import requests
import os

# Localized region options with placeholder
REGIONS = {
    "en": [
        "--select your region--",
        "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen",
        "Hamburg", "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen",
        "Nordrhein Westfalen", "Rheinland Pfalz", "Saarland", "Sachsen",
        "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen"
    ],
    "de": [
        "--bitte Region wählen--",
        "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen",
        "Hamburg", "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen",
        "Nordrhein Westfalen", "Rheinland Pfalz", "Saarland", "Sachsen",
        "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen"
    ]
}
# Enhanced session initialization
def initialize_session():
    """Initialize all session state variables"""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "conversation_started" not in st.session_state:
        st.session_state["conversation_started"] = False
    
    # Always reset these to ensure new user doesn't inherit locked state
    st.session_state["chat_locked"] = False
    st.session_state["post_final_allowed"] = False
    
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    if "region" not in st.session_state:
        st.session_state["region"] = ""
    if "session_ended" not in st.session_state:
        st.session_state["session_ended"] = False

def reset_session():
    """Completely reset session state for new user"""
    old_session_id = st.session_state.get("session_id")
    
    # Call backend to clean up old session
    if old_session_id:
        try:
            API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost")
            API_PORT = os.environ.get("API_PORT", "8001")

            if "localhost" in API_BASE_URL or "127.0.0.1" in API_BASE_URL:
                API_URL = f"{API_BASE_URL}:8000"
            else:
                API_URL = f"{API_BASE_URL}:{API_PORT}"
            
            requests.post(f"{API_URL}/end_session", json={"session_id": old_session_id}, timeout=5)
            print(f"[FRONTEND] Cleaned up backend session: {old_session_id}")
        except Exception as e:
            print(f"[FRONTEND WARNING] Could not clean up backend session: {e}")
    
    # Reset all session state
    st.session_state.clear()
    initialize_session()
    print(f"[FRONTEND] Session reset. New session ID: {st.session_state['session_id']}")

st.set_page_config(page_title="FRIDA", layout="centered")

st.markdown("""
<style>
            
form {
    border: 2px solid #ff914d !important;  /* orange border */
    border-radius: 8px;
    padding: 16px;
    background-color: #1a1f4a;  /* optional: matches background */
}
           
/* Entire app background */
body, [data-testid="stAppViewContainer"] {
    background-color: #1A1F4A !important;
    color: white !important;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}

/* Title 'FRIDA' in black */
h1, h1 span {
    color: #ff914d !important;
}

/* Make subtitle (under FRIDA) white */
h1 + span, h1 + div span {
    color: white !important;
}

/* Make 'Your Session ID' white */
[data-testid="stMarkdownContainer"] p strong, [data-testid="stMarkdownContainer"] code {
    color: white !important;
}

/* Make "Start Your Conversation" white */
h2, h3 {
    color: white !important;
}

/* Chat bubbles */
div[style*="background-color: #dcf8c6"] {
    background-color: #ff914d !important;
    color: #1b1f4b !important;
    font-size: 16px;
    font-weight: 500;
}
div[style*="background-color: #f1f0f0"] {
    background-color: #ffffff !important;
    color: #1b1f4b !important;
    font-size: 16px;
}

/* Text input box */
textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 16px !important;
    border: 1px solid #cccccc !important;
}

/* Button styling */
.stButton>button {
    background-color: #ff914d;
    color: #1b1f4b;
    font-weight: 600;
    border: none;
    border-radius: 6px;
}
.stButton>button:hover {
    background-color: #ffa86c;
    color: #1b1f4b;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div style='display: flex; align-items: center; gap: 20px;'>
  <h1 style='margin: 0; font-size: 62px;'>👧 FRIDA</h1>
  <span style='color: #38b000; font-size: 18px; font-weight: 500;'>
    Radikalisierung früh erkennen und Hilfe anbieten
  </span>
</div>
""", unsafe_allow_html=True)



#st.markdown("<div style='font-size: 18px; color: green;'>Radikalisierung früh erkennen und Hilfe anbieten.</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 17px; color: gray;'>FRIDA - Früherkennung von Radikalisierung, Identifikation und Direkte Assistenz </div>", unsafe_allow_html=True)

# Initialize session
initialize_session()

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
                    bot_reply = response["reply"]
                    st.session_state["messages"].append(("bot", bot_reply))

                    # Check for final message marker
                    FINAL_MARKERS = [
                        "you did the right thing by not looking away",
                        "du hast das richtige getan, indem du nicht weggeschaut hast"
                    ]
                    lower_reply = bot_reply.lower()
                    if any(marker in lower_reply for marker in FINAL_MARKERS):
                        st.session_state["post_final_allowed"] = True

                    # Lock after user sends their one allowed post-final message
                    elif st.session_state.get("post_final_allowed") and not st.session_state.get("chat_locked"):
                        st.session_state["chat_locked"] = True

            except Exception as e:
                st.error(f"Backend error: {e}")
            st.session_state["user_input"] = ""

    st.markdown("""
    💬 Hi, I am FRIDA, here to assist you in investigating any concerns you may have regarding someone exhibiting early indications of radicalization, particularly in the direction of right-wing extremism. I am a sympathetic, anonymous, and nonjudgmental bot.
    """ if st.session_state["lang"] == "en" else """
    💬 Hallo, ich bin FRIDA und bin hier, um Ihnen bei der Untersuchung von Bedenken zu helfen, die Sie möglicherweise hinsichtlich einer Person haben, die erste Anzeichen einer Radikalisierung zeigt, insbesondere in Richtung Rechtsextremismus. Ich bin ein einfühlsamer, anonymer und vorurteilsfreier Bot.
    """)
    
    #border: 3px solid #e63946;

    
    
    for sender, msg in st.session_state["messages"]:
        if sender == "user":
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='
                    background-color: #dcf8c6;
                    padding: 16px;
                    border-radius: 14px;
                    max-width: 80%;
                    color: black;
                    text-align: left;
                    line-height: 1.6;
                '>{msg}</div>
                <img src='https://img.icons8.com/color/48/male-female-user-group.png' style='margin-left: 8px; width: 32px; height: 32px;' />
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                <img src='https://img.icons8.com/3d-fluency/94/technical-support--v3.png' style='margin-right: 8px; width: 32px; height: 32px;' />
                <div style='
                    background-color: #f1f0f0;
                    padding: 16px;
                    border-radius: 14px;
                    max-width: 85%;
                    color: black;
                    text-align: left;
                    line-height: 1.6;
                '>{msg}</div>
            </div>
            """, unsafe_allow_html=True)


    #st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.get("chat_locked", False):
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("", height=80, key="user_input", disabled=False)
            submit = st.form_submit_button("Send", on_click=send_message)
    else:
        st.info("🔒 The conversation has ended. Thank you for your message.")


    # End Conversation button
    if st.button("End Conversation"):
        # Clean up backend session FIRST
        old_session_id = st.session_state.get("session_id")
        if old_session_id:
            try:
                API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost")
                API_PORT = os.environ.get("API_PORT", "8001")
                if "localhost" in API_BASE_URL or "127.0.0.1" in API_BASE_URL:
                    API_URL = f"{API_BASE_URL}:8000"
                else:
                    API_URL = f"{API_BASE_URL}:{API_PORT}"
                
                requests.post(f"{API_URL}/end_session", json={"session_id": old_session_id}, timeout=5)
            except Exception as e:
                print(f"Backend cleanup failed: {e}")
        
        # Then reset frontend
        st.session_state["conversation_started"] = False
        st.session_state["region"] = ""
        st.session_state["messages"] = []
        st.session_state["session_id"] = str(uuid.uuid4())
        st.rerun()