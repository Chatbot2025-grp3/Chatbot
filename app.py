import streamlit as st
import uuid
import requests

st.title("🛡️ Radicalization Detection Support Bot")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []

lang = st.selectbox("Choose language", ["en", "de"])
region = st.text_input("Enter your region (e.g., berlin, nrw, bundesweit)")

user_input = st.text_area("What would you like to talk about?", height=150)

if st.button("Send") and user_input.strip():
    try:
        raw = requests.post("http://backend:8000/chat", json={
            "session_id": st.session_state["session_id"],
            "message": user_input,
            "lang": lang,
            "region": region
    })
        print("🧪 Raw response text:", raw.text)
        response = raw.json()
    except Exception as e:
        st.error(f"❌ Backend error or invalid JSON: {e}")
        st.stop()


    st.session_state["messages"].append(("🧑 You", user_input))
    st.session_state["messages"].append(("🤖 Bot", response["reply"]))

if st.session_state["messages"]:
    st.markdown("### 💬 Conversation History")
    for speaker, msg in st.session_state["messages"]:
        st.markdown(f"**{speaker}:** {msg}")
