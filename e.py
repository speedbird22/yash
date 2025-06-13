import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore, auth
import google.generativeai as genai
from cryptography.hazmat.primitives import serialization
from PIL import Image
import io
import json
import pandas as pd

# Initialize Streamlit app
st.title("Dish Recognition and Menu Matching")

# Function to validate PEM key
def validate_pem_key(key_str, key_name):
    try:
        key_str = key_str.strip().replace('\r\n', '\n')
        if not key_str.startswith("-----BEGIN PRIVATE KEY-----"):
            st.error(f"{key_name} does not start with '-----BEGIN PRIVATE KEY-----'")
            return False
        if not key_str.endswith("-----END PRIVATE KEY-----"):
            st.error(f"{key_name} does not end with '-----END PRIVATE KEY-----'")
            return False
        serialization.load_pem_private_key(key_str.encode('utf-8'), password=None)
        return True
    except Exception as e:
        st.error(f"Invalid PEM key for {key_name}: {str(e)}")
        return False

# Initialize APIs
try:
    if not all(key in st.secrets for key in ["GOOGLE_CLOUD_VISION_CREDENTIALS", "FIREBASE_CREDENTIALS", "GEMINI"]):
        st.error("Missing sections in secrets.toml")
        st.stop()

    # Google Cloud Vision
    vision_credentials_dict = dict(st.secrets["GOOGLE_CLOUD_VISION_CREDENTIALS"])
    required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id", "auth_uri", "token_uri", "universe_domain"]
    missing_keys = [key for key in required_keys if key not in vision_credentials_dict]
    if missing_keys:
        st.error(f"Invalid Google Cloud Vision credentials. Missing keys: {', '.join(missing_keys)}.")
        st.stop()
    if not validate_pem_key(vision_credentials_dict["private_key"], "Google Cloud Vision"):
        st.stop()
    vision_credentials = service_account.Credentials.from_service_account_info(vision_credentials_dict)
    vision_client = vision.ImageAnnotatorClient(credentials=vision_credentials)

    # Firebase
    firebase_credentials_dict = dict(st.secrets["FIREBASE_CREDENTIALS"])
    missing_keys = [key for key in required_keys if key not in firebase_credentials_dict]
    if missing_keys:
        st.error(f"Invalid Firebase credentials. Missing keys: {', '.join(missing_keys)}.")
        st.stop()
    if not validate_pem_key(firebase_credentials_dict["private_key"], "Firebase"):
        st.stop()
    if not firebase_admin._apps:
        firebase_cred = credentials.Certificate(firebase_credentials_dict)
        firebase_admin.initialize_app(firebase_cred)
    db = firestore.client()

    # Gemini
    gemini_api_key = st.secrets["GEMINI"]["api_key"]
    if not gemini_api_key:
        st.error("Gemini API key is empty in secrets.toml.")
        st.stop()
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

except Exception as e:
    st.error(f"Error initializing APIs: {str(e)}")
    st.stop()

# Customer Authentication
st.sidebar.header("Customer Login")
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")
logout_button = st.sidebar.button("Logout")

if 'user' not in st.session_state:
    st.session_state.user = None

if login_button:
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        st.sidebar.success("Logged in successfully!")
    except Exception as e:
        st.sidebar.error(f"Login failed: {str(e)}")

if logout_button and st.session_state.user:
    st.session_state.user = None
    st.sidebar.success("Logged out successfully!")

# Fetch customer profile
def fetch_customer_profile(user_id):
    try:
        doc = db.collection("customers").document(user_id).get()
        if doc.exists:
            return doc.to_dict()
        return {"order_history": [], "dietary_preferences": []}
    except Exception as e:
        st.error(f"Error fetching customer profile: {str(e)}")
        return {"order_history": [], "dietary_preferences": []}

# Save customer profile
def save_customer_profile(user_id, profile):
    try:
        db.collection("customers").document(user_id).set(profile)
    except Exception as e:
        st.error(f"Error saving customer profile: {str(e)}")

# Dietary Preferences Input
st.sidebar.header("Dietary Preferences")
if st.session_state.user:
    user_id = st.session_state.user["localId"]
    profile = fetch_customer_profile(user_id)
    dietary_options = ["Vegan", "Vegetarian", "Gluten-Free", "Keto", "Dairy-Free", "Low-Sugar", "No Preference"]
    selected_preferences = st.sidebar.multiselect(
        "Select Dietary Preferences",
        dietary_options,
        default=profile.get("dietary_preferences", [])
    )
    if st.sidebar.button("Save Preferences"):
        profile["dietary_preferences"] = selected_preferences
        save_customer_profile(user_id, profile)
        st.sidebar.success("Preferences saved!")
else:
    st.sidebar.warning("Please log in to set dietary preferences.")

# Function to detect dish using Google Cloud Vision
def detect_dish(image_content):
    try:
        image = vision.Image(content=image_content)
        response = vision_client.label_detection(image=image)
        labels = response.label_annotations
        dish_labels = [label.description for label in labels][:5]
        if not dish_labels:
            return "Unknown dish"
        prompt = f"Based on the following labels from an image, identify the most likely dish: {', '.join(dish_labels)}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error detecting dish: {str(e)}")
        return None

# Function to fetch menu from Firebase
@st.cache_data(ttl=3600)
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

# Function to generate personalized recommendations
def get_personalized_recommendations(dish_name, menu_items, order_history, dietary_preferences):
    try:
        menu_text = "\n".join([f"- {item['name']}: {item.get('description', '')}, Ingredients: {', '.join(item.get('ingredients', []))}" for item in menu_items])
        history_text = "\n".join([f"- {order['name']}" for order in order_history]) if order_history else "No order history available."
        preferences_text = ", ".join(dietary_preferences) if dietary_preferences else "No dietary preferences specified."
        prompt = f"""
        Given the detected dish '{dish_name}', the customer's order history:
        {history_text}
        and dietary preferences: {preferences_text},
        recommend up to 3 personalized dishes from the following menu that align with the detected dish, dietary preferences, and popular trends (e.g., vegan, gluten-free, low-sugar options). For each recommendation, suggest possible customizations (e.g., ingredient swaps, portion sizes). Provide the dish name, description, and customizations in a clear format.
        Menu:
        {menu_text}
        If no suitable dishes are found, suggest general alternatives based on the dish type and preferences.
        """
        response = gemini_model.generate_content(prompt)
        recommendations = response.text.strip()
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return "No recommendations available due to an error."

# Customize Menu Option
def customize_menu(menu_items, dietary_preferences, portion_size=None, ingredient_swaps=None):
    try:
        filtered_items = []
        for item in menu_items:
            item_preferences = item.get("dietary_tags", [])
            if not dietary_preferences or any(pref.lower() in [tag.lower() for tag in item_preferences]):
                filtered_item = item.copy()
                if portion_size:
                    filtered_item["portion_size"] = portion_size
                if ingredient_swaps:
                    filtered_item["custom_ingredients"] = ingredient_swaps
                filtered_items.append(filtered_item)
        return filtered_items
    except Exception as e:
        st.error(f"Error customizing menu: {str(e)}")
        return []

# Streamlit Tabs for Navigation
tab1, tab2 = st.tabs(["Dish Recognition", "Menu Exploration"])

with tab1:
    st.header("Upload and Match Dish")
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
                
                # Personalized Recommendations
                if st.session_state.user:
                    st.subheader("Personalized Recommendations")
                    profile = fetch_customer_profile(user_id)
                    recommendations = get_personalized_recommendations(
                        dish_name, menu_items, profile.get("order_history", []), profile.get("dietary_preferences", [])
                    )
                    st.markdown(recommendations)
                else:
                    st.warning("Log in to receive personalized recommendations based on your order history and preferences.")
            else:
                st.error("Could not identify the dish. Please try another image.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab2:
    st.header("Explore Menu")
    st.subheader("Customize Your Menu")
    dietary_filter = st.multiselect(
        "Filter by Dietary Preferences",
        ["Vegan", "Vegetarian", "Gluten-Free", "Keto", "Dairy-Free", "Low-Sugar"],
        key="menu_filter"
    )
    portion_size = st.selectbox("Select Portion Size", ["Regular", "Small", "Large"], index=0)
    ingredient_swaps = st.text_input("Ingredient Swaps (e.g., 'replace cheese with avocado')")
    if st.button("Apply Filters"):
        menu_items = fetch_menu()
        customized_menu = customize_menu(menu_items, dietary_filter, portion_size, ingredient_swaps)
        if customized_menu:
            st.subheader("Customized Menu")
            df = pd.DataFrame([
                {
                    "Name": item["name"],
                    "Description": item.get("description", "No description"),
                    "Ingredients": ", ".join(item.get("ingredients", [])),
                    "Portion Size": item.get("portion_size", "Regular"),
                    "Custom Ingredients": item.get("custom_ingredients", "None")
                }
                for item in customized_menu
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No dishes match the selected filters.")
    
    # Visual Exploration
    st.subheader("Browse by Theme")
    theme = st.selectbox("Select Theme", ["Italian", "Mexican", "Asian", "Desserts", "Healthy"])
    if st.button("Explore Theme"):
        menu_items = fetch_menu()
        prompt = f"""
        From the following menu, suggest 3 dishes that fit the '{theme}' theme. Include the dish name, description, and any relevant dietary tags.
        Menu:
        {'\n'.join([f"- {item['name']}: {item.get('description', '')}, Tags: {', '.join(item.get('dietary_tags', []))}" for item in menu_items])}
        """
        try:
            response = gemini_model.generate_content(prompt)
            st.markdown(response.text.strip())
        except Exception as e:
            st.error(f"Error generating themed suggestions: {str(e)}")
