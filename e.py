import streamlit as st
from google.cloud import vision
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
import os
from PIL import Image
import io
import json

# Initialize Streamlit app
st.title("Dish Recognition and Menu Matching")
st.write("Upload an image of a dish, and we'll identify it and check if it's on the menu!")

# Hardcoded credentials (REPLACE WITH YOUR ACTUAL VALUES)
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Replace with your Gemini API key
FIREBASE_CREDENTIALS = {  # Replace with your Firebase service account JSON content
    "type": "service_account",
    "project_id": "your-project-id",
    "private_key_id": "your-private-key-id",
    "private_key": "-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n",
    "client_email": "your-client-email@your-project-id.iam.gserviceaccount.com",
    "client_id": "your-client-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-client-email%40your-project-id.iam.gserviceaccount.com"
}

# Initialize APIs
try:
    # Google Cloud Vision setup (from secrets.toml)
    if "GOOGLE_CLOUD_VISION_CREDENTIALS" not in st.secrets:
        st.error("Google Cloud Vision credentials not found in secrets.toml under [GOOGLE_CLOUD_VISION_CREDENTIALS].")
        st.stop()
    vision_credentials = dict(st.secrets["GOOGLE_CLOUD_VISION_CREDENTIALS"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_vision_credentials.json"
    with open("temp_vision_credentials.json", "w") as f:
        json.dump(vision_credentials, f)
    vision_client = vision.ImageAnnotatorClient()

    # Firebase setup
    if not firebase_admin._apps:
        firebase_cred = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(firebase_cred)
    db = firestore.client()

    # Gemini API setup
    gemini_api_key = GEMINI_API_KEY
    if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
        st.error("Gemini API key not set. Please update GEMINI_API_KEY in the code.")
        st.stop()
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

except Exception as e:
    st.error(f"Error initializing APIs: {str(e)}")
    st.stop()
finally:
    # Clean up temporary Vision credentials file
    if os.path.exists("temp_vision_credentials.json"):
        os.remove("temp_vision_credentials.json")

# Function to detect dish using Google Cloud Vision
def detect_dish(image_content):
    try:
        image = vision.Image(content=image_content)
        response = vision_client.label_detection(image=image)
        labels = response.label_annotations
        dish_labels = [label.description for label in labels if "food" in label.description.lower() or "dish" in label.description.lower()]
        
        if not dish_labels:
            return "Unknown dish"
        
        # Use Gemini to refine the dish identification
        prompt = f"Based on the following labels from an image, identify the most likely dish: {', '.join(dish_labels)}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error detecting dish: {str(e)}")
        return None

# Function to fetch menu from Firebase
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
            return None, "No menu items found in the database."
        
        # Create a prompt for Gemini to match the dish
        menu_text = "\n".join([f"- {item['name']}: {item.get('description', '')}" for item in menu_items])
        prompt = f"""
        Given the dish '{dish_name}', find the most similar or exact match from the following menu:
        {menu_text}
        Return the name of the matching dish or suggest a similar one if no exact match is found.
        If no close match exists, return 'No close match found'.
        """
        response = gemini_model.generate_content(prompt)
        match = response.text.strip()
        
        # Check if an exact or close match was found
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
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Dish", use_column_width=True)

    # Convert image to bytes for Vision API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_content = img_byte_arr.getvalue()

    # Detect dish
    with st.spinner("Identifying the dish..."):
        dish_name = detect_dish(img_content)
    
    if dish_name:
        st.write(f"Detected dish: **{dish_name}**")
        
        # Fetch menu and find matching dish
        with st.spinner("Checking menu for matches..."):
            menu_items = fetch_menu()
            match, message = find_matching_dish(dish_name, menu_items)
        
        # Display results
        st.subheader("Menu Match Result")
        st.write(message)
        if match:
            st.write(f"**Dish Name**: {match['name']}")
            st.write(f"**Description**: {match.get('description', 'No description available')}")
            st.write(f"**Ingredients**: {', '.join(match.get('ingredients', []))}")
    else:
        st.error("Could not identify the dish. Please try another image.")
