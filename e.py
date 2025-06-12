import streamlit as st
import os
import json
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

# Google Cloud Vision service account JSON
google_vision_credentials = {
    "type": "service_account",
    "project_id": "halogen-premise-462605-t3",
    "private_key_id": "23c74c4ff326cdb76ac4ac76d83f8ebf73042420",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCS7hfRU1GZYN90\neeKRmOHbqHkZcehVgH/rKRqt/RDM4oIT8sm/bWdnFGmhq5synCnNaVyRSzCZ0ajY\nxLhtxSVJQlnM2TTRAX53ZL/k8c4Gr8WF66BCrlZYVyjElDWbhtiCDnPsIbaBRdbS\nj7d8eqelClLWZgCmWMlyiHeu99Poqo5bfPnNWpVTxd1dlSbLkefgzr2iFKgXEyS6\nrLthUn/J0OT2vb7VUPpjbysuMWXyPKQLcJf7P060QCtS/uXvPlxjH0wXhAuPqMwC\nv+KuHuCrmS8nsvkMgvW88A4sazXX5yUxohx7EJLfl6esYkhGvXelh4G2CELUpL8d\nwCWIkToPAgMBAAECggEAEqm8i8G2ETHtYEmlHlJZigG11CYUWVv3o6K2eVg2PJge\n5rLFrpOSeVTt7/OwxHZylt1lvSzcUQWBGXuvY34RtOMBLhR+8XdrnXXGLCoMveuw\nvsiFtEKZxCHaE1IyFv3DXY3BasKDgJsi5N8NcvnPdpa/m+0b8wU/HVW5QhfHqnti\nTp0yePyqQwXgAc9w3As5LioDXsL+ObrsY86dOZK0DSRb5lRfup0zXrEn8QgOHHyf\nG9SlsLpIUqhiPXNPRFOSVflg1E8im39ep0gFf5sS8ddnNdzJHS+FkD+MmgrYIp1I\nFQufe2DgWs/RtVCyFvnEgq1y3Al5CyKbgpJLJU8npQKBgQDGMvjAffg+0qo339By\nd6J3m9qlUI8oWmw+WxNboAAgWpfrjIht23hlLqn9NfCgnaPvEfhSNaqoTXM+WXNV\nAeE5T2QqpluaJIX6VYnlGWH/xJ2HVh0IcvrKKFOcdAqExy9RPPI4UtrwO/b4z0fl\nK08b/ptuUNFX4XSwtWRGl7cOhQKBgQC9x4HeXnbLzeiM0m6Pp+cZAycBPtveN6hj\nQK1grAceCzfGDV+Fv28CeTGnvofOEBpy/wsldGUHLowx0c+xGwlQ2dUnQ42PZfCE\nvozL6WHRi0CxizFw6nQFBhmglZkh8akXZs28XPhOtaSyFqzFvafJ9pp4/K0c53tR\nnKjiUglcgwKBgQC+fMC86JUNdBUq7E3/peGdCUrD9cARLY65A3mAZy+X6Nn4BjId\nO4Dj5kx0U0I4bCnhnKjIAlJJvV5Uf04sVkkrdpUztH13kUC/DzUf3hxk4IfySZ5P\nv4ovf0CUrqZpZLKiZmv6HA/WMhu0mCtmIxC+PB8QkFYTG1m2eC+u9SN3rQKBgHCA\nDkqF8SJLw+mG7SeXvvUi20JvwuLuV5HOy3idQfLbb4qmbtPYcbQjp/3qgpLzFNrK\nug7P+vvQ6ia2W8p/XnxhRWxrLUWuhnAKjWhrxyLyy7zz6LEpKvG3dgWt6QMoQLaw\nqJFbA4+VOjagHndyQD8HQvcpwm16A66TkcHoI8iFAoGAVwEohiFXayHvk1D3hGNd\n+WIcXhi96F7Yp1uK9+SpMO3/7IjhtG656dADKQ2+5XhoJ7VmztUb19AYwTf22OEG\nP0hh2pYBYK6Asi73oeOpZxRScdZMFiN4SDBUjZa8NUsR/9RKw1wM6zhORsQcakED\nsU6GxdoOLwjQL4AsX2rvO7w=\n-----END PRIVATE KEY-----\n",
    "client_email": "restaurant@halogen-premise-462605-t3.iam.gserviceaccount.com",
    "client_id": "109108467055426880660",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/restaurant%40halogen-premise-462605-t3.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Firebase Admin SDK service account JSON
firebase_credentials = {
    "type": "service_account",
    "project_id": "restaurant-data-backend",
    "private_key_id": "9636792e28585d67a1ed327cdc27c41fa88c3754",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDcjQwyvhocV5Ah\niC17UYkqjMOtkdjLuywCj6YKP+kLPrIBLUQKa5/PQGXt0QxmpxDiGrlJtNovUIdR\nv1gKIk2wxWGop4/9lEJtH4YvpAYInX3xvtg6YJRYlN+R0Zxh1OZn6jDhjHSwqmfv\nbe/pK7sLcI+/uxncCyJ8jyaeIpcZT4qqkzuxQ8mmbDMTOeZWrEHEzy4vxui5Py2M\nWLApBCw7a3ps2bBHADIv90jnEO2HwXSh2UX3iy/6KhOy43WEdDEfEpMEk3bM4u91\n9TleSAvxGnCCFSrjipJ6MrwSov4aD7KFv1vJzZT9QNqKl9XiuKCz8cLr8phXOEV0\nhUgkejjnAgMBAAECggEARv6ANrVK4mStWJ3lRhTw+mllc7HG/424lPp4kEQSWDRO\nSGKxzEjooGYyaWMpgsG0hZPkoP0+XoylgoL9bAWuzIA893U4vH/FAitrnlpGNu+7\nYt3z7Ja6KemCLgYzOKq3oCuaoH/98ABqhH/3Ai+5fe012Jn5sQNEjqPl4nFwAg0n\njXAO0esUxpLfDHZ83f1pfLgdfOKI6BwMGqBHV4zfS8L471VNJYeOxcRcTCz7a5o0\nSRmR2h9Xnnz4iJOvTocC+EEyoHGVGbc00EqLuEoOKDqMQ55i75jj+ghL7z/XNrfk\npZTBN7nKsLZWjwUkqDOdg0zEfpC3rUPAyJLVKOwKgQKBgQD4X6Vzl/c42XzTmZTw\nHlIVsb7G4j+MJCJdrGIUe8M3n2L7RpeHofesvQTcjSGe53ZDqSQus92YsE2ulb3Y\nrFZOwis9F38RXmMtzD9vSnHWRyKM1JgIpubJvn4q3iJwoqyOgW1a2QHUbu555xaf\nb5+TnzQLplXZG4sOoSG4mOb2HwKBgQDjUrMw0v12FVxsM2NqqWw8Qs7ZBRtWoOVs\n3o0O6a72DtnwOD1Y9MamGbFB67vDNk+oiLP8t3rk4bxF+x4jkH35q8d1XC9qDSww\nPkAUaAg5hPjnGQVznUj4juZL/hdvAC8QoauGqZmIUPRpsv4cLFK+jnOld0EHGMiE\nb+9iKc0UOQKBgH0UTdIEua+bd01ojqTN4DCkrpqh3bbJi7T41vvRx+H/Fm3Mgwr+\n+ie3mPco68GGdvxj9aC/W91FDBnbtxuizmQjTHsblhY9Hl01+swlBWcPs8qQVXAl\n/RukHw2fiGCIy7WIYHXbyxwcMWSah74LDKXfCurC/YC0ajcX1k+MUOOpAoGBAIqw\negLftyPEBI8/CviYRSC+4dQl+Xfw0giJ/yWKDOSySuT0avlK3aeZJTxCxltjV1ZL\nkQQuLRxXXLaVbt3j2ffphddkVsktIDiOwimxDtOI/RKBgYH4A/0hsf/LFiDyy8Qp\n3qxZ7QHt4jBatA4cPJ9l6ciZ6WKbDDtAz5vkROqBAoGAJGhjmLHF81KDtfd1I1rt\nXxqomfL/RN5j6P5MfChzgwnmgg9iCXczFNS22oX+PbgSMAg873RU9uusAVdKfoYP\nvrtWSvxA5g0DwcsZD4TvQUmWmK3AX8/g18LnV/+cQIaSZus7raMmMUpZ+SC6bVXM\nVXKijo9mm59rRloFCENY6ZU=\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-fbsvc@restaurant-data-backend.iam.gserviceaccount.com",
    "client_id": "115855681795792637429",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40restaurant-data-backend.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Debug: Log Vision credentials info (non-sensitive fields)
st.write("Debug: Vision Service Account Info")
st.write(f"Project ID: {google_vision_credentials.get('project_id')}")
st.write(f"Client Email: {google_vision_credentials.get('client_email')}")
st.write(f"Token URI: {google_vision_credentials.get('token_uri')}")
logger.debug("Vision credentials: %s", {k: v for k, v in google_vision_credentials.items() if k != 'private_key'})

# Initialize Google Vision client with embedded credentials
vision_client = None
try:
    vision_credentials = service_account.Credentials.from_service_account_info(
        google_vision_credentials,
        scopes=['https://www.googleapis.com/auth/cloud-vision']
    )
    # Debug: Test token generation
    vision_credentials.refresh(Request())
    st.write("Debug: Vision API token generated successfully")
    logger.debug("Vision API token: %s", vision_credentials.token[:10] + "...")
    vision_client = ImageAnnotatorClient(credentials=vision_credentials)
except Exception as e:
    st.error(f"Error initializing Vision client: {str(e)}")
    st.write("Debug: Ensure Vision API is enabled in project 'halogen-premise-462605-t3' and service account has 'roles/vision.apiUser'.")
    st.write("Debug: Check billing account and API quotas in Google Cloud Console.")
    logger.error("Vision client initialization failed: %s", str(e))
    vision_client = None

# Initialize Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBRy86kB4ASDYgO0dttEOBYvocY6n13ii4")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    gemini_model = GenerativeModel('gemini-1.5-flash')
    st.write("Debug: Gemini API key loaded successfully")
    logger.debug("Gemini API key loaded")
else:
    st.error("Gemini API key not found. Please add GEMINI_API_KEY to .env file.")
    gemini_model = None
    logger.error("Gemini API key missing")

# Initialize Firebase Firestore with embedded credentials
try:
    firebase_cred = credentials.Certificate(firebase_credentials)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(firebase_cred)
    db = firestore.client()
    menu_ref = db.collection('menu')
    st.write("Debug: Firestore initialized successfully")
    logger.debug("Firestore initialized")
except Exception as e:
    st.error(f"Error initializing Firestore: {str(e)}")
    logger.error("Firestore initialization failed: %s", str(e))
    menu_ref = None

# Function to detect dish using Google Vision
def detect_dish(image_file):
    if vision_client is None:
        raise Exception("Vision client not initialized")
    try:
        content = image_file.read()
        image = vision.Image(content=content)
        response = vision_client.label_detection(image=image)
        if response.error.message:
            logger.error("Vision API error: %s", response.error.message)
            raise Exception(f"Vision API error: {response.error.message}")
        labels = [label.description for label in response.label_annotations]
        logger.debug("Vision API labels: %s", labels)
        return labels
    except Exception as e:
        logger.error("Dish detection failed: %s", str(e))
        raise

# Function to get Gemini suggestions
def get_gemini_suggestions(dish_labels):
    if gemini_model is None:
        raise Exception("Gemini model not initialized")
    try:
        prompt = f"Based on the dish labels {dish_labels}, provide a detailed recipe for the dish or suggest similar dishes with brief descriptions."
        response = gemini_model.generate_content([prompt])
        logger.debug("Gemini response: %s", response.text[:100] + "...")
        return response.text
    except Exception as e:
        logger.error("Gemini suggestion failed: %s", str(e))
        raise

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
        logger.debug("Firestore matches: %d", len(matching_dishes))
        return pd.DataFrame(matching_dishes)
    except Exception as e:
        logger.error("Firestore query failed: %s", str(e))
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

# Run the app
if __name__ == "__main__":
    st.write("Upload an image to start!")
