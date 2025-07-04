import streamlit as st
import uuid

# Configure page
st.set_page_config(page_title="FRIDA", layout="wide")

# Initialize session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Title
st.title("👧 FRIDA")
st.subheader("Radikalisierung früh erkennen und Hilfe anbieten")

# Show Session ID clearly
st.info(f"**Your Session ID:** {st.session_state.session_id}")

# Simple input
user_message = st.text_area("Type your message here:", height=100)

# Columns for button on the right
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col5:
    if st.button("📤 SEND", use_container_width=True):
        if user_message:
            st.success(f"Message sent: {user_message}")
        else:
            st.warning("Please type a message first!")

# Test info
st.write("---")
st.write("🧪 **Test Status:** Basic FRIDA is working!")
st.write("✅ Session ID visible")
st.write("✅ Input field working") 
st.write("✅ Send button on the right")