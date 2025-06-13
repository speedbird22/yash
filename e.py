from cryptography.hazmat.primitives import serialization
import streamlit as st
vision_key = st.secrets["GOOGLE_CLOUD_VISION_CREDENTIALS"]["private_key"].strip()
firebase_key = st.secrets["FIREBASE_CREDENTIALS"]["private_key"].strip()
try:
    serialization.load_pem_private_key(vision_key.encode('utf-8'), password=None)
    st.write("Vision private key is valid")
except Exception as e:
    st.error(f"Vision private key error: {str(e)}")
try:
    serialization.load_pem_private_key(firebase_key.encode('utf-8'), password=None)
    st.write("Firebase private key is valid")
except Exception as e:
    st.error(f"Firebase private key error: {str(e)}")
