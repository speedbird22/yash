import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from cryptography.hazmat.primitives import serialization
from PIL import Image
import io

# Custom CSS for a complex, textured UI inspired by the image
st.markdown("""
    <style>
    /* Dark gradient background with a subtle texture effect */
    .stApp {
        background: linear-gradient(135deg, #1a0033 0%, #330066 50%, #003366 100%);
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
    }
    /* Textured container for main content */
    .main-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(5px);
        margin: 20px 0;
    }
    /* Header styling with neon glow */
    .header {
        background: linear-gradient(90deg, #ff00cc, #3333ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 0, 204, 0.7), 0 0 20px rgba(51, 51, 255, 0.7);
        margin-bottom: 20px;
    }
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(135deg, #330066, #003366);
        color: #e0e0e0;
    }
    .stSidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    /* Button styling with neon effect and hover animation */
    .stButton>button {
        background: linear-gradient(45deg, #ff00cc, #3333ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 0 10px rgba(255, 0, 204, 0.5), 0 0 20px rgba(51, 51, 255, 0.5);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #ff33cc, #6666ff);
        box-shadow: 0 0 15px rgba(255, 0, 204, 0.8), 0 0 30px rgba(51, 51, 255, 0.8);
        transform: scale(1.05);
    }
    /* Result box with a textured, glowing border */
    .result-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 0, 204, 0.3);
        box-shadow: 0 0 10px rgba(255, 0, 204, 0.3);
        margin-top: 10px;
    }
    /* Headings with neon glow */
    h3, h4 {
        color: #ff66cc;
        text-shadow: 0 0 5px rgba(255, 102, 204, 0.5);
    }
    /* Spinner text color */
    .stSpinner {
        color: #ff66cc;
    }
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255, 0, 204, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for instructions
with st.sidebar:
    st.header("📋 How to Use This App")
    st.markdown("""
    1. **Upload an Image**: Use the file uploader below to upload a JPG or PNG image of a dish.
    2. **Wait for Analysis**: The app will identify the dish and search for a matching item in the menu.
    3. **View Results**: Check the results below the image for the detected dish and menu match.
    - Supported formats: JPG, PNG
    - Ensure the image is clear for accurate detection.
    """)

# Header
st.markdown('<div class="header">🍽️ Dish Recognition and Menu Matching</div>', unsafe_allow_html=True)

# Main content container
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

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

    # Streamlit file uploader with improved layout
    st.markdown("### 📸 Upload Your Dish Image")
    uploaded_file = st.file_uploader("Choose a JPG or PNG image of the dish", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            if image.format not in ["JPEG", "PNG"]:
                st.error("Unsupported image format. Please upload a JPG or PNG image.")
                st.stop()

            # Create two columns for image and results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Uploaded Image")
                st.image(image, caption="Your Dish", use_container_width=True)

            with col2:
                st.markdown("#### Analysis Results")
                with st.spinner("🔍 Identifying the dish..."):
                    dish_name = detect_dish(image.tobytes())
                if dish_name:
                    st.success(f"Detected dish: **{dish_name}**")
                    with st.spinner("🍴 Checking menu for matches..."):
                        menu_items = fetch_menu()
                        match, message = find_matching_dish(dish_name, menu_items)
                    
                    # Display results in a styled box
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("**Menu Match Result**")
                    st.write(message)
                    if match:
                        st.markdown(f"**Dish Name**: {match['name']}")
                        st.markdown(f"**Description**: {match.get('description', 'No description available')}")
                        st.markdown(f"**Ingredients**: {', '.join(match.get('ingredients', []))}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("Could not identify the dish. Please try another image.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
