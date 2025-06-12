import streamlit as st
import os
from dotenv import load_dotenv
from google.cloud import vision
from google.cloud.vision_v1 import ImageAnnotatorClient
from google.generativeai import GenerativeModel
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image
import io
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize Google Vision client with service account
google_cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "halogen-premise-462605-t3-23c74c4ff326.json")
vision_client = ImageAnnotatorClient.from_service_account_json(google_cred_path)

# Initialize Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBRy86kB4ASDYgO0dttEOBYvocY6n13ii4")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    gemini_model = GenerativeModel('gemini-1.5-flash')
else:
    st.error("Gemini API key not found. Please add GEMINI_API_KEY to .env file.")

# Initialize Firebase Firestore
firebase_cred_path = os.getenv("FIREBASE_CREDENTIALS", "restaurant-data-backend-firebase-adminsdk-fbsvc-9636792e28.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cred_path)
    firebase_admin.initialize_app(cred)
db = firestore.client()
menu_ref = db.collection('menu')

# Function to detect dish using Google Vision
def detect_dish(image_file):
    content = image_file.read()
    image = vision.Image(content=content)
    response = vision_client.label_detection(image=image)
    labels = [label.description for label in response.label_annotations]
    return labels

# Function to get Gemini suggestions
def get_gemini_suggestions(dish_labels):
    prompt = f"Based on the dish labels {dish_labels}, provide a detailed recipe for the dish or suggest similar dishes with brief descriptions."
    response = gemini_model.generate_content([prompt])
    return response.text

# Function to search Firestore menu
def search_firebase_menu(dish_labels):
    try:
        docs = menu_ref.stream()
        matching_dishes = []
        for doc in docs:
            dish = doc.to_dict()
            dish_name = dish.get('name', '').lower()
            ingredients = [ing.lower() for ing in dish.get('ingredients', [])]
            if any(label.lower() in dish_name or any(label.lower() in ing for ing in ingredients) for label in dish_labels):
                matching_dishes.append({
                    'Name': dish.get('name'),
                    'Ingredients': ', '.join(dish.get('ingredients', [])),
                    'Category': dish.get('category')
                })
        return pd.DataFrame(matching_dishes)
    except Exception as e:
        st.error(f"Error querying Firestore: {str(e)}")
        return pd.DataFrame()

# Streamlit app
st.set_page_config(page_title="Dish Recognition App")
st.header("Dish Recognition and Recipe Suggestion")

# Image upload
uploaded_file = st.file_uploader("Upload an image of a dish...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Dish", use_column_width=True)

    # Detect dish
    st.subheader("Detected Dish Labels")
    try:
        dish_labels = detect_dish(uploaded_file)
        st.write(dish_labels)
    except Exception as e:
        st.error(f"Error detecting dish: {str(e)}")
        dish_labels = []

    # Get Gemini suggestions
    if dish_labels and gemini_api_key:
        st.subheader("Recipe or Similar Dish Suggestions")
        with st.spinner("Fetching suggestions..."):
            try:
                gemini_response = get_gemini_suggestions(dish_labels)
                st.markdown(gemini_response)
            except Exception as e:
                st.error(f"Error fetching Gemini suggestions: {str(e)}")

    # Search Firestore menu
    if dish_labels:
        st.subheader("Matching Menu Items")
        menu_df = search_firebase_menu(dish_labels)
        if not menu_df.empty:
            st.dataframe(menu_df, use_container_width=True)
        else:
            st.write("No matching dishes found in the menu.")

# Run the app
if __name__ == "__main__":
    st.write("Upload an image to start!")
