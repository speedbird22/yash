import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from cryptography.hazmat.primitives import serialization
from PIL import Image
import io
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import random
import string

# Streamlit configuration
st.set_page_config(page_title="Dish Recognition App", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background: linear-gradient(to bottom, #1e1e2f, #2a2a3d); color: #e0e0e0; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; }
    .stFileUploader { border: 2px dashed #4CAF50; padding: 10px; border-radius: 10px; }
    h1, h2, h3 { color: #4CAF50; font-family: 'Arial', sans-serif; }
    .stDataFrame { border: 1px solid #4CAF50; border-radius: 10px; }
    .game-box {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(to bottom, #4B0082, #8A2BE2);
        color: #e0e0e0;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 0 15px #BA55D3;
        z-index: 1000;
        font-family: 'Cinzel', serif;
    }
    .word-search-button {
        width: 40px;
        height: 40px;
        margin: 2px;
        font-size: 16px;
        border-radius: 5px;
        border: 1px solid #4CAF50;
        background-color: #2a2a3d;
        color: #e0e0e0;
    }
    .word-search-button.found {
        background-color: #4CAF50;
        color: white;
    }
    .word-search-button.selected {
        background-color: #BA55D3;
        color: white;
    }
    .word-search-button.hint {
        background-color: #FFD700;
        color: #1e1e2f;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Streamlit app
st.title("🍽️ Dish Recognition and Menu Matching")

# Validate PEM key
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
    gemini_api_key = st.secrets["GEMINI"]["api_key"]
    if not gemini_api_key:
        st.error("Gemini API key is empty in secrets.toml.")
        st.stop()
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Error initializing APIs: {str(e)}")
    st.stop()

# Dietary Preferences
st.sidebar.header("Dietary Preferences")
dietary_options = ["Vegan", "Vegetarian", "Gluten-Free", "Keto", "Dairy-Free", "Low-Sugar", "No Preference"]
selected_preferences = st.sidebar.multiselect(
    "Select Dietary Preferences",
    dietary_options,
    default=["No Preference"],
    key="sidebar_dietary"
)

# Word Search Logic
def generate_word_search(size=10, words=None):
    if not words:
        return None, None
    grid = [[random.choice(string.ascii_uppercase) for _ in range(size)] for _ in range(size)]
    word_positions = []
    directions = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]
    occupied_cells = set()
    for word in words:
        word = word.upper()
        placed = False
        attempts = 0
        max_attempts = 50
        while not placed and attempts < max_attempts:
            attempts += 1
            start_row = random.randint(0, size-1)
            start_col = random.randint(0, size-1)
            direction = random.choice(directions)
            dr, dc = direction
            word_len = len(word)
            end_row = start_row + dr * (word_len - 1)
            end_col = start_col + dc * (word_len - 1)
            if 0 <= end_row < size and 0 <= end_col < size:
                can_place = True
                temp_positions = []
                for i in range(word_len):
                    r = start_row + i * dr
                    c = start_col + i * dc
                    cell = (r, c)
                    if cell in occupied_cells:
                        # Allow overlap only if the letter matches
                        if grid[r][c] != word[i]:
                            can_place = False
                            break
                    temp_positions.append(cell)
                if can_place:
                    positions = []
                    for i in range(word_len):
                        r = start_row + i * dr
                        c = start_col + i * dc
                        grid[r][c] = word[i]
                        positions.append((r, c))
                        occupied_cells.add((r, c))
                    word_positions.append((word, positions))
                    placed = True
        if not placed:
            return None, None
    return grid, word_positions

def initialize_word_search():
    food_words = [
        ["Pizza", "Sushi", "Pasta", "Tacos", "Salad", "Burger", "Soup", "Curry"],
        ["Noodle", "Rice", "Steak", "Fries", "Cake", "Donut", "Bread", "Tofu"],
        ["Ramen", "Chili", "Salsa", "Bagel", "Pancake", "Waffle", "Pie", "Kebab"],
        ["Lasagna", "Sandwich", "Omelet", "Wrap", "Smoothie", "Muffin", "Scone", "Fish"],
        ["Quinoa", "Tart", "Crepe", "Bacon", "Sorbet", "Gyoza", "Shrimp", "Rice"]
    ]
    grids = []
    for words in food_words:
        grid, positions = generate_word_search(10, words)
        if grid and positions:
            grids.append((grid, words, positions))
    if not grids:
        return None, None, None
    return random.choice(grids)

# Initialize word search
if "word_search_initialized" not in st.session_state:
    st.session_state.word_search_initialized = True
    st.session_state.grid, st.session_state.words, st.session_state.positions = initialize_word_search()
    st.session_state.found_words = []
    st.session_state.selected_cells = []
    st.session_state.show_game = False
    st.session_state.stars = 0
    st.session_state.hint_cells = []

# Floating Game Box
if st.button("Hey, Wanna Play a Food Word Search? 🌌", key="game_toggle"):
    st.session_state.show_game = not st.session_state.show_game

if st.session_state.show_game:
    st.markdown('<div class="game-box">', unsafe_allow_html=True)
    st.markdown("### Food Word Search")
    st.markdown("Click letters to form words (horizontal, vertical, or diagonal). Find all to earn 5 stars!")
    
    # Display grid
    for i in range(len(st.session_state.grid)):
        cols = st.columns(10)
        for j in range(len(st.session_state.grid[i])):
            cell_key = f"cell_{i}_{j}"
            is_found = any((i, j) in pos for _, pos in st.session_state.positions if any(w for w, _ in st.session_state.positions if w in st.session_state.found_words))
            is_selected = (i, j) in st.session_state.selected_cells
            is_hint = (i, j) in st.session_state.hint_cells
            button_style = "word-search-button"
            if is_found:
                button_style += " found"
            elif is_hint:
                button_style += " hint"
            elif is_selected:
                button_style += " selected"
            with cols[j]:
                if st.button(st.session_state.grid[i][j], key=cell_key):
                    if (i, j) not in st.session_state.selected_cells:
                        st.session_state.selected_cells.append((i, j))
                    else:
                        st.session_state.selected_cells.remove((i, j))
    
    # Check for found words
    def check_word_formed():
        selected = sorted(st.session_state.selected_cells)
        if len(selected) < 2:
            return
        dr = selected[1][0] - selected[0][0] if len(selected) > 1 else 0
        dc = selected[1][1] - selected[0][1] if len(selected) > 1 else 0
        is_line = all(
            (selected[i+1][0] - selected[i][0] == dr and selected[i+1][1] - selected[i][1] == dc)
            for i in range(len(selected)-1)
        )
        if is_line:
            word = "".join(st.session_state.grid[r][c] for r, c in selected)
            for w, pos in st.session_state.positions:
                if word.upper() == w.upper():
                    if w not in st.session_state.found_words:
                        st.session_state.found_words.append(w)
                        st.session_state.selected_cells = []
                        st.balloons()
                    break
    
    check_word_formed()
    
    # Display words
    st.markdown("**Words to Find**:")
    word_cols = st.columns(8)
    for idx, word in enumerate(st.session_state.words):
        with word_cols[idx]:
            color = "green" if word.upper() in st.session_state.found_words else "#e0e0e0"
            st.markdown(f"<span style='color: {color}'>{word}</span>", unsafe_allow_html=True)
    
    # Calculate stars
    found_count = len(st.session_state.found_words)
    total_words = len(st.session_state.words)
    if found_count > 0 or st.session_state.stars > 0:
        stars = max(0, 5 - (total_words - found_count))
        if stars > st.session_state.stars:
            st.session_state.stars = stars
        st.markdown(f"**Stars Earned**: {'★' * st.session_state.stars + '☆' * (5 - st.session_state.stars)}")
        if found_count == total_words:
            st.balloons("🎉 You found all words! 5 stars!")
    
    # Hint button
    if st.button("Hint (-1 Star)"):
        if st.session_state.stars > 0:
            unfound_words = [(w, pos) for w, pos in st.session_state.positions if w not in st.session_state.found_words]
            if unfound_words:
                word, positions = random.choice(unfound_words)
                hint_cell = random.choice(positions)
                st.session_state.hint_cells.append(hint_cell)
                st.session_state.stars = max(0, st.session_state.stars - 1)
                st.rerun()
    
    # Reset game
    if st.button("New Game"):
        st.session_state.grid, st.session_state.words, st.session_state.positions = initialize_word_search()
        st.session_state.found_words = []
        st.session_state.selected_cells = []
        st.session_state.hint_cells = []
        st.session_state.stars = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Detect dish
def detect_dish(image_content):
    def _detect_dish():
        try:
            image = vision.Image(content=image_content)
            response = vision_client.label_detection(image=image)
            labels = [label.description for label in response.label_annotations][:5]
            if not labels:
                return "Unknown dish"
            prompt = f"Based on the following labels from an image, identify the most likely dish: {', '.join(labels)}"
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error detecting dish: {str(e)}"
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_detect_dish)
        try:
            return future.result(timeout=10)
        except TimeoutError:
            return "Dish detection timed out. Please try again."

# Fetch menu
@st.cache_data(ttl=3600)
def fetch_menu():
    try:
        menu_ref = db.collection("menu")
        docs = menu_ref.stream()
        menu_items = [{"id": doc.id, **doc.to_dict()} for doc in docs]
        if not menu_items:
            st.warning("No menu items found in Firebase.")
            return []
        return menu_items
    except Exception as e:
        st.error(f"Error fetching menu: {str(e)}")
        return []

# Find matching dish
def find_matching_dish(dish_name, menu_items):
    try:
        if not menu_items:
            return None, "No menu items found in the database."
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

# Personalized recommendations
def get_personalized_recommendations(dish_name, menu_items, dietary_preferences):
    try:
        menu_text = "\n".join([f"- {item['name']}: {item.get('description', '')}, Ingredients: {', '.join(item.get('ingredients', []))}, Tags: {', '.join(item.get('dietary_tags', []))}" for item in menu_items])
        preferences_text = ", ".join(dietary_preferences) if dietary_preferences else "No dietary preferences specified."
        prompt = f"""
        Given the detected dish '{dish_name}' and dietary preferences: {preferences_text},
        recommend up to 3 personalized dishes from the following menu that align with the detected dish, dietary preferences, and popular trends. For each, suggest customizations. Provide output in a markdown list with dish name, description, dietary tags, and customizations.
        Menu:
        {menu_text}
        If no suitable dishes are found, suggest general alternatives.
        """
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return "No recommendations available due to an error."

# Customize menu
def customize_menu(menu_items, dietary_preferences, portion_size=None, ingredient_swaps=None):
    try:
        filtered_items = []
        for item in menu_items:
            item_preferences = item.get("dietary_tags", [])
            if not dietary_preferences or "No Preference" in dietary_preferences or any(p.lower() in [tag.lower() for tag in item_preferences] for p in dietary_preferences):
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

# Tabs
tab1, tab2 = st.tabs(["Dish Recognition", "Menu Exploration"])

with tab1:
    st.header("Upload and Match Dish")
    uploaded_file = st.file_uploader("Upload an image of the dish (JPG or PNG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            if image.format not in ["JPEG", "PNG"]:
                st.error("Unsupported image format. Please upload a JPG or PNG image.")
                st.stop()
            st.image(image, caption="Uploaded Dish", use_container_width=True)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_content = img_byte_arr.getvalue()
            with st.spinner("Identifying the dish..."):
                dish_name = detect_dish(img_content)
            if isinstance(dish_name, str) and not dish_name.startswith("Error"):
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
                    st.write(f"**Dietary Tags**: {', '.join(match.get('dietary_tags', []))}")
                st.subheader("Recommended Dishes")
                with st.spinner("Generating personalized recommendations..."):
                    recommendations = get_personalized_recommendations(dish_name, menu_items, selected_preferences)
                st.markdown("### Suggested Menu")
                st.markdown(recommendations)
                recommended_items = customize_menu(menu_items, selected_preferences)
                if recommended_items:
                    df = pd.DataFrame([
                        {
                            "Name": item["name"],
                            "Description": item.get("description", "No description"),
                            "Ingredients": ", ".join(item.get("ingredients", [])),
                            "Dietary Tags": ", ".join(item.get("dietary_tags", [])),
                            "Portion Size": item.get("portion_size", "Regular"),
                            "Custom Ingredients": item.get("custom_ingredients", "None")
                        }
                        for item in recommended_items
                    ])
                    st.markdown("### Menu Preview")
                    st.dataframe(df, use_container_width=True)
            else:
                st.error(dish_name if dish_name.startswith("Error") else "Could not identify the dish. Please try another image.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab2:
    st.header("Explore Menu")
    st.subheader("Customize Your Menu")
    dietary_filter = st.multiselect(
        "Filter by Dietary Preferences",
        ["Vegan", "Vegetarian", "Gluten-Free", "Keto", "Dairy-Free", "Low-Sugar", "No Preference"],
        default=[pref for pref in selected_preferences if pref in ["Vegan", "Vegetarian", "Gluten-Free", "Keto", "Dairy-Free", "Low-Sugar", "No Preference"]],
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
                    "Dietary Tags": ", ".join(item.get("dietary_tags", [])),
                    "Portion Size": item.get("portion_size", "Regular"),
                    "Custom Ingredients": item.get("custom_ingredients", "None")
                }
                for item in customized_menu
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No dishes match the selected filters.")
    st.subheader("Browse by Theme")
    theme = st.selectbox("Select Theme", ["Italian", "Mexican", "Asian", "Desserts", "Healthy"])
    if st.button("Explore Theme"):
        menu_items = fetch_menu()
        prompt = f"""
        From the following menu, suggest 3 dishes that fit the '{theme}' theme and align with the dietary preferences: {', '.join(dietary_filter) if dietary_filter else 'None'}. Include the dish name, description, and dietary tags in a formatted markdown list.
        Menu:
        {'\n'.join([f"- {item['name']}: {item.get('description', '')}, Tags: {', '.join(item.get('dietary_tags', []))}" for item in menu_items])}
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: gemini_model.generate_content(prompt).text.strip())
            try:
                response = future.result(timeout=10)
                st.markdown("### Themed Suggestions")
                st.markdown(response)
            except TimeoutError:
                st.error("Themed suggestions timed out. Please try again.")
            except Exception as e:
                st.error(f"Error generating themed suggestions: {str(e)}")
