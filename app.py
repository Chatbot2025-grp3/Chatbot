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

st.set_page_config(page_title="FRIDA", layout="centered")

# Session initialization
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "conversation_started" not in st.session_state:
    st.session_state["conversation_started"] = False
if "chat_locked" not in st.session_state:
    st.session_state["chat_locked"] = False
if "post_final_allowed" not in st.session_state:
    st.session_state["post_final_allowed"] = False
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "region" not in st.session_state:
    st.session_state["region"] = ""

# Set default theme colors (dark theme)
background_color = "#001f3f"
text_color = "white"
chat_bg_color = "#f1f0f0"

# Enhanced CSS with right-aligned button styling
st.markdown(f"""
<style>
/* CSS Custom Properties for theme adaptation */
:root {{
    --primary-bg: {background_color};
    --primary-text: {text_color};
    --accent-color: #ff914d;
    --accent-hover: #ffa86c;
    --accent-text: #1b1f4b;
    --input-bg: #ffffff;
    --input-text: #000000;
    --input-border: #cccccc;
    --code-bg: #2d3748;
    --code-text: #e2e8f0;
    --form-border: #ff914d;
    --chat-bg: {chat_bg_color};
}}

/* Form styling with better visibility */
form {{
    border: 2px solid var(--form-border) !important;
    border-radius: 8px;
    padding: 16px;
    background-color: var(--primary-bg);
    box-shadow: 0 2px 8px rgba(255, 145, 77, 0.2);
}}

/* Entire app background */
body, [data-testid="stAppViewContainer"] {{
    background-color: var(--primary-bg) !important;
    color: var(--primary-text) !important;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}}

/* Title 'FRIDA' styling */
h1, h1 span {{
    color: var(--accent-color) !important;
}}

/* Make subtitle (under FRIDA) white */
h1 + span, h1 + div span {{
    color: var(--primary-text) !important;
}}

/* Make section headers white */
h2, h3 {{
    color: var(--primary-text) !important;
}}

/* FIXED: Selectbox labels - force white color with maximum specificity */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] > div > label,
.stSelectbox label,
.stSelectbox > div > label,
div.stSelectbox label,
div.stSelectbox > div > label,
.row-widget.stSelectbox label,
.row-widget.stSelectbox > div > label {{
    color: white !important;
    font-weight: 500 !important;
}}

/* Additional broad targeting for all labels */
label,
label *,
div label,
div label *,
.stApp label,
.stApp label *,
[data-testid="stApp"] label,
[data-testid="stApp"] label * {{
    color: white !important;
}}

/* Ultra-specific targeting for selectbox elements */
div[data-testid="stSelectbox"] > label > div,
div[data-testid="stSelectbox"] > label > div > p,
div[data-testid="stSelectbox"] label p,
div[data-testid="stSelectbox"] label span,
div[data-testid="stSelectbox"] label div {{
    color: white !important;
}}

/* Force all text in selectbox containers to be white */
div[data-testid="stSelectbox"] * {{
    color: white !important;
}}

/* Override any markdown or paragraph styling in labels */
div[data-testid="stSelectbox"] p,
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] div {{
    color: white !important;
}}

/* Chat bubbles with better contrast */
div[style*="background-color: #dcf8c6"] {{
    background-color: var(--accent-color) !important;
    color: var(--accent-text) !important;
    font-size: 16px;
    font-weight: 500;
}}
div[style*="background-color: #f1f0f0"] {{
    background-color: var(--chat-bg) !important;
    color: var(--accent-text) !important;
    font-size: 16px;
}}

/* Text input box with better visibility */
textarea {{
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    font-size: 16px !important;
    border: 2px solid var(--input-border) !important;
    border-radius: 6px !important;
    padding: 12px !important;
}}

textarea:focus {{
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 2px rgba(255, 145, 77, 0.2) !important;
}}

/* Button styling with enhanced visibility */
.stButton>button {{
    background-color: var(--accent-color) !important;
    color: var(--accent-text) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(255, 145, 77, 0.3) !important;
}}

.stButton>button:hover {{
    background-color: var(--accent-hover) !important;
    color: var(--accent-text) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(255, 145, 77, 0.4) !important;
}}

/* Form submit button specific styling - RIGHT ALIGNED */
form button[kind="formSubmit"] {{
    background-color: #ff914d !important;
    color: #1b1f4b !important;
    font-weight: 700 !important;
    border: 2px solid #ff914d !important;
    min-width: 120px !important;
    min-height: 45px !important;
    border-radius: 6px !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
    margin-top: 10px !important;
}}

form button[kind="formSubmit"]:hover {{
    background-color: #ffa86c !important;
    border-color: #ffa86c !important;
    color: #1b1f4b !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(255, 145, 77, 0.4) !important;
}}

/* Disabled button styling */
form button[kind="formSubmit"]:disabled {{
    background-color: #cccccc !important;
    color: #666666 !important;
    border-color: #cccccc !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* Alternative targeting for form buttons */
div[data-testid="stForm"] button {{
    background-color: #ff914d !important;
    color: #1b1f4b !important;
    font-weight: 700 !important;
    border: 2px solid #ff914d !important;
    min-width: 120px !important;
    min-height: 45px !important;
    border-radius: 6px !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
}}

div[data-testid="stForm"] button:hover {{
    background-color: #ffa86c !important;
    border-color: #ffa86c !important;
    color: #1b1f4b !important;
}}

div[data-testid="stForm"] button:disabled {{
    background-color: #cccccc !important;
    color: #666666 !important;
    border-color: #cccccc !important;
    cursor: not-allowed !important;
}}

/* Right-aligned button container */
.button-container-right {{
    display: flex;
    justify-content: flex-end;
    margin-top: 10px;
}}

/* Selectbox styling */
div[data-testid="stSelectbox"] > div > div {{
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    border: 2px solid var(--input-border) !important;
}}

/* Selectbox dropdown options styling */
div[data-testid="stSelectbox"] select {{
    background-color: var(--input-bg) !important;
    color: black !important;
}}

/* Selectbox selected value styling */
div[data-testid="stSelectbox"] > div > div > div {{
    color: black !important;
}}

/* Target the actual select element and its options */
div[data-testid="stSelectbox"] select option {{
    color: black !important;
    background-color: white !important;
}}

/* Additional targeting for dropdown content */
div[data-testid="stSelectbox"] div[role="listbox"] {{
    color: black !important;
}}

div[data-testid="stSelectbox"] div[role="option"] {{
    color: black !important;
}}

/* Force all text inside selectbox dropdown to be black */
div[data-testid="stSelectbox"] > div > div * {{
    color: black !important;
}}

/* Warning message styling - REMOVED background and border */
div[data-testid="stAlert"] {{
    background-color: transparent !important;
    color: #00ff99 !important;
    border: none !important;
    padding: 0 !important;
    font-weight: 600 !important;
}}

/* Info message styling */
div[data-testid="stInfo"] {{
    background-color: #dbeafe !important;
    color: #1e40af !important;
    border: 1px solid #3b82f6 !important;
}}

/* Force white text for conversation ended message */
div[data-testid="stInfo"] *,
.stInfo *,
.stInfo {{
    color: white !important;
}}

/* Override any default info styling */
div[data-testid="stInfo"] p,
div[data-testid="stInfo"] span,
div[data-testid="stInfo"] div {{
    color: white !important;
}}

/* High contrast mode support */
@media (prefers-contrast: high) {{
    :root {{
        --input-border: #000000;
        --code-bg: #000000;
        --code-text: #ffffff;
    }}
    
    .stButton>button {{
        border: 2px solid var(--accent-text) !important;
    }}
}}
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

st.markdown("<div style='font-size: 17px; color: gray;'>FRIDA - Früherkennung von Radikalisierung, Identifikation und Direkte Assistenz </div>", unsafe_allow_html=True)

if (
    not st.session_state["conversation_started"]
    and st.session_state["lang"] is not None
    and st.session_state["region"] not in ["", REGIONS[st.session_state["lang"]][0]]
):
    st.session_state["conversation_started"] = True

# Language and region setup
if not st.session_state["conversation_started"]:
    st.markdown(f"<span style='color: #00ff99; font-weight: 600;'>Your Session ID: {st.session_state['session_id']}</span>", unsafe_allow_html=True)
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
    st.markdown(f"<span style='color: #00ff99; font-weight: 600;'>Your Session ID: {st.session_state['session_id']}</span>", unsafe_allow_html=True)
    
    new_lang = st.selectbox("🌐 Language", ["English", "Deutsch"], index=0 if st.session_state["lang"] == "en" else 1)
    new_lang_code = "en" if new_lang == "English" else "de"
    if new_lang_code != st.session_state["lang"]:
        st.session_state["lang"] = new_lang_code
        st.session_state["messages"].append(("bot", f"🌐 Language has been changed to {'English' if new_lang_code == 'en' else 'Deutsch'}."))

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

                    # Enhanced final message detection
                    FINAL_MARKERS = [
                        "you did the right thing by not looking away",
                        "du hast das richtige getan, indem du nicht weggeschaut hast",
                        # Additional markers for more robust detection
                        "zentrum-demokratische-kultur.de",
                        "thank you for sharing your concerns with me",
                        "danke, dass sie ihre sorgen mit mir geteilt haben"
                    ]
                    
                    lower_reply = bot_reply.lower()
                    
                    # Check if this is the final conclusion message
                    if any(marker in lower_reply for marker in FINAL_MARKERS):
                        st.session_state["chat_locked"] = True
                        st.session_state["post_final_allowed"] = False

            except Exception as e:
                st.error(f"Backend error: {e}")
            st.session_state["user_input"] = ""

    st.markdown("""
    💬 Hi, I am FRIDA, here to assist you in investigating any concerns you may have regarding someone exhibiting early indications of radicalization, particularly in the direction of right-wing extremism. I am a sympathetic, anonymous, and nonjudgmental bot.
    """ if st.session_state["lang"] == "en" else """
    💬 Hallo, ich bin FRIDA und bin hier, um Ihnen bei der Untersuchung von Bedenken zu helfen, die Sie möglicherweise hinsichtlich einer Person haben, die erste Anzeichen einer Radikalisierung zeigt, insbesondere in Richtung Rechtsextremismus. Ich bin ein einfühlsamer, anonymer und vorurteilsfreier Bot.
    """)

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
                    background-color: {chat_bg_color};
                    padding: 16px;
                    border-radius: 14px;
                    max-width: 85%;
                    color: black;
                    text-align: left;
                    line-height: 1.6;
                '>{msg}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chat input form with conditional disable
    if not st.session_state.get("chat_locked", False):
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("Type your message here...", height=80, key="user_input")
            
            # Right-aligned button using columns
            col1, col2 = st.columns([4, 1])
            with col2:
                submit = st.form_submit_button("Send", on_click=send_message, use_container_width=True)
    else:
        # Show disabled form when chat is locked
        with st.form(key="chat_form_disabled", clear_on_submit=False):
            user_input = st.text_area("Type your message here...", height=80, key="user_input_disabled", disabled=True, placeholder="Conversation has ended.")
            
            # Right-aligned disabled button
            col1, col2 = st.columns([4, 1])
            with col2:
                submit = st.form_submit_button("Send", disabled=True, use_container_width=True)
        
        st.info("🔒 The conversation has ended. Thank you for your message." if st.session_state["lang"] == "en" else "🔒 Das Gespräch ist beendet. Vielen Dank für Ihre Nachricht.")

    if st.button("End Conversation"):
        st.session_state["conversation_started"] = False
        st.session_state["region"] = ""
        st.session_state["messages"] = []
        st.session_state["chat_locked"] = False
        st.session_state["post_final_allowed"] = False
        st.session_state["session_id"] = str(uuid.uuid4())
        st.rerun()