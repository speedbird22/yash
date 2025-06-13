import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from cryptography.hazmat.primitives import serialization
from PIL import Image
import io
import signal
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom CSS for enhanced UI features without colors
st.markdown("""
    <style>
    /* Reset any previous background colors */
    .stApp {
        background: none;
        font-family: 'Arial', sans-serif;
    }
    /* Sidebar styling without colors */
    .stSidebar {
        background: none;
    }
    /* File uploader enhancements */
    .stFileUploader {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #999;
        background: rgba(0, 0, 0, 0.05);
        transform: scale(1.02);
    }
    /* Button animation for file uploader */
    .stFileUploader button {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stFileUploader button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    /* Custom loading animation for processing */
    .custom-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #999;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    /* Card-like container for output */
    .output-card {
        background: #fff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
        opacity: 0;
        animation: fadeIn 0.5s ease forwards;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    /* Remove any previous color-related styles for headers and text */
    h3, h4 {
        color: inherit;
        text-shadow: none;
    }
    .stSpinner {
        color: inherit;
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
st.markdown("### 🍽️ Dish Recognition and Menu Matching")

# Timeout handler for API calls
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Function to resize image
def resize_image(image, max_size=(800, 800)):
    logger.info("Resizing image...")
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

# Main content container
with st.container():
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
            logger.info("Starting dish detection...")
            image = vision.Image(content=image_content)
            # Set timeout for Vision API call
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 seconds timeout
            logger.info("Calling Vision API for label detection...")
            response = vision_client.label_detection(image=image)
            labels = response.label_annotations
            signal.alarm(0)  # Disable timeout
            logger.info("Vision API call completed.")
            
            dish_labels = [label.description for label in labels][:5]
            if not dish_labels:
                return "Unknown dish"
            
            # Set timeout for Gemini API call
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 seconds timeout
            logger.info("Calling Gemini API for dish identification...")
            prompt = f"Based on the following labels from an image, identify the most likely dish: {', '.join(dish_labels)}"
            response = gemini_model.generate_content(prompt)
            signal.alarm(0)  # Disable timeout
            logger.info("Gemini API call completed.")
            
            return response.text.strip()
        except TimeoutError:
            logger.error("Dish detection timed out.")
            st.error("Dish detection took too long. Please try again with a different image.")
            return None
        except Exception as e:
            logger.error(f"Error detecting dish: {str(e)}")
            st.error(f"Error detecting dish: {str(e)}")
            return None

    # Function to fetch menu from Firebase
    @st.cache_data(ttl=3600)
    def fetch_menu():
        try:
            logger.info("Fetching menu from Firebase...")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 seconds timeout
            menu_ref = db.collection("menu")
            docs = menu_ref.stream()
            menu_items = [{"id": doc.id, **doc.to_dict()} for doc in docs]
            signal.alarm(0)  # Disable timeout
            logger.info("Menu fetched successfully.")
            return menu_items
        except TimeoutError:
            logger.error("Fetching menu timed out.")
            st.error("Fetching menu took too long. Please try again later.")
            return []
        except Exception as e:
            logger.error(f"Error fetching menu: {str(e)}")
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
            # Set timeout for Gemini API call
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 seconds timeout
            logger.info("Calling Gemini API for menu matching...")
            response = gemini_model.generate_content(prompt)
            signal.alarm(0)  # Disable timeout
            logger.info("Gemini API call for menu matching completed.")
            
            match = response.text.strip()
            for item in menu_items:
                if item["name"].lower() == match.lower():
                    return item, "Exact match found!"
                if match.lower() in item["name"].lower() or match.lower() in item.get("description", "").lower():
                    return item, "Similar dish found!"
            return None, match if match != "No close match found" else "No close match found in the menu."
        except TimeoutError:
            logger.error("Menu matching timed out.")
            st.error("Menu matching took too long. Please try again.")
            return None, "Operation timed out."
        except Exception as e:
            logger.error(f"Error matching dish: {str(e)}")
            st.error(f"Error matching dish: {str(e)}")
            return None, "Error occurred while matching dish."

    # Streamlit file uploader with improved features
    st.markdown("### 📸 Upload Your Dish Image")
    uploaded_file = st.file_uploader("Choose a JPG or PNG image of the dish", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            logger.info("Image uploaded, starting processing...")
            # Open and resize image
            image = Image.open(uploaded_file)
            if image.format not in ["JPEG", "PNG"]:
                st.error("Unsupported image format. Please upload a JPG or PNG image.")
                st.stop()
            
            image = resize_image(image)  # Resize image to speed up processing
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_content = img_byte_arr.getvalue()

            # Create two columns for image and results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Uploaded Image")
                st.image(image, caption="Your Dish", use_container_width=True)

            with col2:
                st.markdown("#### Analysis Results")
                with st.spinner(""):
                    # Custom spinner with animation
                    st.markdown('<div class="custom-spinner"></div>Identifying the dish...', unsafe_allow_html=True)
                    dish_name = detect_dish(img_content)
                if dish_name:
                    # Output in a styled card with fade-in animation
                    st.markdown('<div class="output-card">', unsafe_allow_html=True)
                    st.markdown(f"**Detected Dish**: {dish_name}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.spinner(""):
                        # Custom spinner with animation
                        st.markdown('<div class="custom-spinner"></div>Checking menu for matches...', unsafe_allow_html=True)
                        menu_items = fetch_menu()
                        match, message = find_matching_dish(dish_name, menu_items)
                    
                    # Display results in a styled card with fade-in animation
                    st.markdown('<div class="output-card">', unsafe_allow_html=True)
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
            logger.error(f"Error processing image: {str(e)}")
            st.error(f"Error processing image: {str(e)}")
