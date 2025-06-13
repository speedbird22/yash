import streamlit as st
import os
from google.cloud import vision
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from PIL import Image
import io

# Retrieve Google Cloud Vision credentials from Streamlit secrets
google_creds_dict = {
    "type": st.secrets["google_credentials"]["type"],
    "project_id": st.secrets["google_credentials"]["project_id"],
    "private_key_id": st.secrets["google_credentials"]["private_key_id"],
    "private_key": st.secrets["google_credentials"]["private_key"],
    "client_email": st.secrets["google_credentials"]["client_email"],
    "client_id": st.secrets["google_credentials"]["client_id"],
    "auth_uri": st.secrets["google_credentials"]["auth_uri"],
    "token_uri": st.secrets["google_credentials"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["google_credentials"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["google_credentials"]["client_x509_cert_url"],
    "universe_domain": st.secrets["google_credentials"]["universe_domain"]
}
google_creds = service_account.Credentials.from_service_account_info(google_creds_dict)

# Initialize Google Vision client with the credentials
vision_client = vision.ImageAnnotatorClient(credentials=google_creds)

# Set up Firebase credentials using the JSON file
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-credentials.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Set up Gemini API key
GEMINI_API_KEY = "your-gemini-api-key"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Streamlit app title
st.title("Dish Recognition and Menu Matching")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image of a dish", type=["jpg", "jpeg", "png"])

def detect_dish(image_content):
    """Use Google Vision API to detect labels in the image."""
    image = vision.Image(content=image_content)
    response = vision_client.label_detection(image=image)
    labels = response.label_annotations
    return [label.description for label in labels]

def get_menu_from_firebase():
    """Fetch menu items from Firestore."""
    menu_ref = db.collection("menu")
    docs = menu_ref.stream()
    menu_items = []
    for doc in docs:
        item = doc.to_dict()
        item["id"] = doc.id
        menu_items.append(item)
    return menu_items

def find_matching_dish(dish_labels, menu_items):
    """Use Gemini API to find matching or similar dishes in the menu."""
    menu_text = "\n".join([f"{item['name']}: {item['description']}" for item in menu_items])
    prompt = f"""
    Based on the following detected labels from an image: {', '.join(dish_labels)},
    find the most matching or similar dish from this menu:
    {menu_text}
    
    Return the name of the matching dish or suggest a similar dish if no exact match is found.
    If no match is found, return 'No match found'.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Dish Image", use_column_width=True)

    # Convert image to bytes for Vision API
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_content = image_bytes.getvalue()

    # Detect dish using Google Vision API
    with st.spinner("Detecting dish..."):
        dish_labels = detect_dish(image_content)
        st.write("Detected Labels:", dish_labels)

    # Fetch menu from Firebase
    with st.spinner("Fetching menu..."):
        menu_items = get_menu_from_firebase()

    # Find matching dish using Gemini API
    with st.spinner("Finding matching dish..."):
        matching_dish = find_matching_dish(dish_labels, menu_items)
        st.success(f"Matching Dish: {matching_dish}")

# Instructions for running the app
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload an image of a dish.")
st.sidebar.write("2. The app will detect the dish using Google Vision API.")
st.sidebar.write("3. It will then check for matching or similar items in the menu using Gemini API.")
st.sidebar.write("4. Ensure Google credentials are set in Streamlit Cloud secrets.")
st.sidebar.write("5. Ensure firebase-credentials.json is uploaded to Streamlit Cloud.")
