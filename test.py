import streamlit as st

st.title("🧪 Streamlit Test")
st.write("Hello! If you can see this text, Streamlit is working perfectly!")
st.success("✅ Great! Now we can build your FRIDA app.")

if st.button("Click me to test"):
    st.balloons()
    st.write("🎉 Button works too!")