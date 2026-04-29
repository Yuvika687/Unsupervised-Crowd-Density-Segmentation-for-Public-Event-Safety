"""Minimal test to check if Streamlit renders at all."""
import streamlit as st

st.set_page_config(page_title="Test", layout="wide")
st.title("TEST PAGE")
st.write("If you see this, Streamlit is working.")

# End of minimal test script
