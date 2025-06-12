import streamlit as st
import os
from google.cloud import vision
from google.cloud.vision_v1 import ImageAnnotatorClient
from google.generativeai import GenerativeModel
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from PIL import Image
import io
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set page config as the first Streamlit command
st.set_page_config(page_title="Dish Recognition App")

# Firebase Admin SDK credentials (hardcoded)
firebase_credentials = {
    "type": "service_account",
    "project_id": "restaurant-data-backend",
    "private_key_id": "9636792e28585d67a1ed327cdc27c41fa88c3754",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDcjQwyvhocV5Ah\niC17UYkqjMOtkdjLuywCj6YKP+kLPrIBLUQKa5/PQGXt0QxmpxDiGrlJtNovUIdR\nv1gKIk2wxWGop4/9lEJtH4YvpAYInX3xvtg6YJRYlN+R0Zxh1OZn6jDhjHSwqmfv\nbe/pK7sLcI+/uxncCyJ8jyaeIpcZT4qqkzuxQ8mmbDMTOeZWrEHEzy4vxui5Py2M\nWLApBCw7a3ps2bBHADIv90jnEO2HwXSh2UX3iy/6KhOy43WEdDEfEpMEk3bM4u91\n9TleSAvxGnCCFSrjipJ6MrwSov4aD7KFv1vJzZT9QNqKl9XiuKCz8cLr8phXOEV0\nhUgkejjnAgMBAAECggEARv6ANrVK4mStWJ3lRhTw+mllc7HG/424lPp4kEQSWDRO\nSGKxzEjooGYyaWMpgsG0hZPkoP0+XoylgoL9bAWuzIA893U4vH/FAitrnlpGNu+7\nYt3z7Ja6KemCLgYzOKq3oCuaoH/98ABqhH/3Ai+5fe012Jn5sQNEjqPl4nFwAg0n\njXAO0esUxpLfDHZ83f1pfLgdfOKI6BwMGqBHV4zfS8L471VNJYeOxcRcTCz7a5o0\nSRmR2h9Xnnz4iJOvTocC+EEyoHGVGbc00EqLuEoOKDqMQ55i75jj+ghL7z/XNrfk\npZTBN7nKsLZWjwUkqDOdg0zEfpC3rUPAyJLVKOwKgQKBgQD4X6Vzl/c42XzTmZTw\nHlIVsb7G4j+MJCJdrGIUe8M3n2L7RpeHofesvQTcjSGe53ZDqSQus92YsE2ulb3Y\nrFZOwis9F38RXmMtzD9vSnHWRyKM1JgIpubJvn4q3iJwoqyOgW1a2QHUbu555xaf\nb5+TnzQLplXZG4sOoSG4mOb2HwKBgQDjUrMw0v12FVxsM2NqqWw8Qs7ZBRtWoOVs\n3o0O6a72DtnwOD1Y9MamGbFB67vDNk+oiLP8t3rk4bxF+x4jkH35q8d1XC9qDSww\nPkAUaAg5hPjnGQVznUj4juZL/hdvAC8QoauGqZmIUPRpsv4cLFK+jnOld0EHGMiE\nb+9iKc0UOQKBgH0UTdIEua+bd01ojqTN4DCkrpqh3bbJi7T41vvRx+H/Fm3Mgwr+\n+ie3mPco68GGdvxj9aC/W91FDBnbtxuizmQjTHsblhY9Hl01+swlBWcPs8qQVXAl\n/RukHw2fiGCIy7WIYHXbyxwcMWSah74LDKXfCurC/YC0ajcX1k+MUOOpAoGBAIqw\negLftyPEBI8/CviYRSC+4dQl+Xfw0giJ/yWKDOSySuT0avlK3aeZJTxCxltjV1ZL\nkQQuLRxXXLaVbt3j extender7ffphddkVsktIDiOwimxDtOI/RKBgYH4A/0hsf/LFiDyy8Qp\n3qxZ7QHt4jBatA4cPJ9l6ciZ6WKbDDtAz5ez7ROqBAoGAJGhjmLHF\n81KDtfd1I1rt\nXxqomfL/RN5j6P5MfChzgwnmgg9iCXczFNS22oX+PbgSMAg873RU9uus\nAVdKfoYP\nvrtWSvxA5g0DwvHsZD4TvQUmWmK3AX8/g18LnV/+cQIaSZus7ra6Mm\nMUpZ+SC6bVXM\nVXKijo9mm59nRloFCENY6ZU=\n\n-----END PRIVATE KEY-----\n    "client_email": "firebase-adminsdk-fbsvc@restaurant-data-backend.iam.gserviceaccount.com",\n    "client_id": "115855681795792637429",\n    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40restaurant-data-backend.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Initialize Firebase
try:
    firebase_cred = credentials.Certificate(firebase_credentials)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(firebase_cred)
    db = firestore.client()
    menu_ref = db.collection('menu")
    logger.debug("Firestore initialized")
except Exception as e:
    st.sidebar.error(f"Error initializing Firestore: {str(e)}")
    logger.error(f"Firestore initialization failed: {str(e)}")
    menu_ref = None

# Initialize Gemini API
try:
    gemini_api_key = "AIZaSyBRyBRy86kB4ASDYgO0dttEOBYvocY6n13ii4" # Hardcoded
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    gemini_model = GenerativeModel('gemini-1.5-flash')
    logger.debug("Gemini API key loaded successfully")
except Exception as e:
    st.sidebar.error(f"Error initializing Gemini: {str(e)}")
    st.sidebar.write("Debug: The Gemini API key may be invalid. Generate a new key in Google Cloud Console.")
    logger.error(f"Gemini initialization failed: {str(e)}")
    gemini_model = None

# Initialize Google Vision client
try:
    vision_credentials = service_account.Credentials.from_service_account_info(
        st.secrets["google_vision_credentials"],
        scopes=['https://www.googleapis.com/auth/cloud-vision']
    )
    vision_credentials.refresh(Request())
    logger.debug(f"Vision API token: {vision_credentials.token[:10]}...")
    vision_client = ImageAnnotatorClient(credentials=vision_credentials)
    logger.debug("Vision API client initialized")
except KeyError as e:
    st.sidebar.error("Google Vision credentials not found in Streamlit secrets. Add google_vision_credentials to secrets.toml.")
    logger.error(f"Vision credentials missing: {str(e)}")
    vision_client = None
except Exception as e:
    st.sidebar.error(f"Error initializing Vision client: {str(e)}")
    st.sidebar.write(f"Debug: Ensure Vision API is enabled in project 'halogen-premise-462605-t3' and service account has 'roles/vision.apiUser'.")
    st.sidebar.write(f"Debug: Verify billing account and API quotas in Google Cloud Console.")
    logger.error(f"Vision client initialization failed: {str(e)}")
    vision_client = None

# Debug: Display initialization info in sidebar
with st.sidebar:
    st.header("Debug Information")
    if "google_vision_credentials" in st.secrets:
        st.write("**Google Vision API**")
        st.write(f"Project ID: {st.secrets['google_vision_credentials'].get('project_id')}")
        st.write(f"Client Email: {st.secrets['google_vision_credentials'].get('client_email')}")
        st.write(f"Token URI: {st.secrets['google_vision_credentials'].get('token_uri')}")
    if vision_client:
        st.write("Debug: Vision API client initialized successfully")
    if gemini_model:
        st.write("Debug: Gemini API key loaded successfully")
    if menu_ref:
        st.write("Debug: Firestore initialized successfully")

# Function to detect dish using Google Vision
def detect_dish(image_file):
    if vision_client is None:
        raise Exception("Vision client not initialized")
    try:
        content = image_file.read()
        image = vision.Image(content=image_content)
        response = vision_client.label_detection(image=image_image)
        if response.error_message:
            logger.error(f"Vision API error: {response.error_message}")
            raise Exception(f"Vision API error: {response.error_message}")
        labels = [label.description.strip() for label in response.label_annotations]
        logger.debug(f"Vision API labels: {labels}")
        return labels
    except Exception as e:
        logger.error(f"Dish detection failed: {str(e)}")
        raise

# Function to get Gemini suggestions
def get_gemini_suggestions(dish_labels):
    if gemini_model is None:
        raise Exception("Gemini model not initialized")
    try:
        prompt = f"Based on the dish labels {dish_labels}, provide a detailed recipe for the dish or suggest similar dishes with brief descriptions."
        response = gemini_model.generate_content([prompt])
        logger.debug(f"Gemini response: {response.text[:500]}...")
        return response.text
    except Exception as e:
        logger.error(f"Gemini suggestion failed: {str(e)}")
        return str(e)

# Function to search Firestore menu
def search_firebase_menu(dish_labels):
    if menu_ref is None:
        raise Exception("Firestore client not initialized")
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
        logger.debug(f"Firestore matches: {len(matching_dishes)}")
        return pd.DataFrame(matching_dishes)
    except Exception as e:
        logger.error(f"Firestore query failed: {str(e)}")
        st.error(f"Error querying Firestore: {str(e)}")
        return pd.DataFrame()

# Streamlit app
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
    if dish_labels and gemini_model:
        st.subheader("Recipe or Similar Dish Suggestions")
        with st.spinner("Fetching suggestions..."):
            try:
                gemini_response = get_gemini_suggestions(dish_labels)
                st.markdown(gemini_response)
            except Exception as e:
                st.error(f"Error fetching Gemini suggestions: {str(e)}")

    # Search Firestore menu
    if dish_labels and menu_ref:
        st.subheader("Matching Menu Items")
        menu_df = search_firebase_menu(dish_labels)
        if not menu_df.empty:
            st.dataframe(menu_df, use_container_width=True)
        else:
            st.write("No matching dishes found in the menu.")

# Footer
st.write("Upload an image to start!")
