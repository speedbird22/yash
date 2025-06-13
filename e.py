import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image
import io

# Initialize Streamlit app
st.title("Dish Recognition and Menu Matching")
st.write("Upload an image of a dish, and we'll identify it and check if it's on the menu!")

# Initialize APIs
try:
    # Validate secrets
    if not all(key in st.secrets for key in ["GOOGLE_CLOUD_VISION_CREDENTIALS", "FIREBASE_CREDENTIALS", "GEMINI"]):
        st.error("Missing sections in secrets.toml: Ensure GOOGLE_CLOUD_VISION_CREDENTIALS, FIREBASE_CREDENTIALS, and GEMINI are defined.")
        st.stop()

    # Debug: Confirm secrets are loaded
    st.write("DEBUG: Secrets sections found:", list(st.secrets.keys()))

    # Google Cloud Vision setup
    vision_credentials_dict = dict(st.secrets["GOOGLE_CLOUD_VISION_CREDENTIALS"])
    required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id", "auth_uri", "token_uri", "universe_domain"]
    missing_keys = [key for key in required_keys if key not in vision_credentials_dict]
    if missing_keys:
        st.error(f"Invalid Google Cloud Vision credentials. Missing keys: {', '.join(missing_keys)}.")
        st.stop()
    st.write("DEBUG: Vision credentials keys:", list(vision_credentials_dict.keys()))
    vision_credentials = service_account.Credentials.from_service_account_info(vision_credentials_dict)
    vision_client = vision.ImageAnnotatorClient(credentials=vision_credentials)

    # Firebase setup
    firebase_credentials_dict = dict(st.secrets["FIREBASE_CREDENTIALS"])
    missing_keys = [key for key in required_keys if key not in firebase_credentials_dict]
    if missing_keys:
        st.error(f"Invalid Firebase credentials. Missing keys: {', '.join(missing_keys)}.")
        st.stop()
    st.write("DEBUG: Firebase credentials keys:", list(firebase_credentials_dict.keys()))
    if not firebase_admin._apps:
        firebase_cred = credentials.Certificate(firebase_credentials_dict)
        firebase_admin.initialize_app(firebase_cred)
    db = firestore.client()

    # Gemini API setup
    gemini_api_key = st.secrets["GEMINI"]["api_key"]
    if not gemini_api_key:
        st.error("Gemini API key is empty in secrets.toml.")
        st.stop()
    st.write("DEBUG: Gemini API key loaded:", bool(gemini_api_key))
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

except Exception as e:
    st.error(f"Error initializing APIs: {str(e)}")
    st.stop()

# Function to detect dish using Google Cloud Vision
def detect_dish(image_content):
    try:
        image = vision.Image(content=image_content)
        response = vision_client.label_detection(image=image)
        labels = response.label_annotations
        dish_labels = [label.description for label in labels][:5]  # Take top 5 labels
        if not dish_labels:
            return "Unknown dish"
        prompt = f"Based on the following labels from an image, identify the most likely dish: {', '.join(dish_labels)}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error detecting dish: {str(e)}")
        return None

# Function to fetch menu from Firebase
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_menu():
    try:
        menu_ref = db.collection("menu")
        docs = menu_ref.stream()
        menu_items = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        return menu_items
    except Exception as e:
        st.error(f"Error fetching menu: {str(e)}")
        return []

# Function to find matching or similar dishes using Gemini
def find_matching_dish(dish_name, menu_items):
    try:
        if not menu_items:
            return None, "No menu items found in the database. Please contact the restaurant admin."
        menu_text = "\n".join([f"- {item['name']}: {item.get('description', '')}" for item in menu_items])
        prompt = f"""
        Given the dish '{dish_name}', find the most similar or exact match from the following menu:
        {menu_text}
        Return the name of the matching dish or suggest a similar one if no exact match is found.
        If no close match exists, return 'No close match found'.
        """
        response = gemini_model.generate_content(prompt)
        match = response.text.strip()
        for item in menu_items:
            if item["name"].lower() == match.lower():
                return item, "Exact match found!"
            if match.lower() in item["name"].lower() or match.lower() in item.get("description", "").lower():
                return item, "Similar dish found!"
        return None, match if match != "No close match found" else "No close match found in the menu."
    except Exception as e:
        st.error(f"Error matching dish: {str(e)}")
        return None, "Error occurred while matching dish."

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload an image of the dish", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        if image.format not in ["JPEG", "PNG"]:
            st.error("Unsupported image format. Please upload a JPG or PNG image.")
            st.stop()
        st.image(image, caption="Uploaded Dish", use_column_width=True)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_content = img_byte_arr.getvalue()
        with st.spinner("Identifying the dish..."):
            dish_name = detect_dish(img_content)
        if dish_name:
            st.write(f"Detected dish: **{dish_name}**")
            with st.spinner("Checking menu for matches..."):
                menu_items = fetch_menu()
                match, message = find_matching_dish(dish_name, menu_items)
            st.subheader("Menu Match Result")
            st.write(message)
            if match:
                st.write(f"**Dish Name**: {match['name']}")
                st.write(f"**Description**: {match.get('description', 'No description available')}")
                st.write(f"**Ingredients**: {', '.join(match.get('ingredients', []))}")
        else:
            st.error("Could not identify the dish. Please try another image.")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
