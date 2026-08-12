import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import zipfile
import io
import urllib.request
from colorizer import ImageColorizer, generate_histogram_plot

# Set page config for a widescreen layout
st.set_page_config(
    page_title="Chromatic AI - Deep Image Colorizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directory Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")

PROTO_PATH = os.path.join(MODELS_DIR, "colorization_deploy_v2.prototxt")
MODEL_PATH = os.path.join(MODELS_DIR, "colorization_release_v2.caffemodel")
HULL_PATH = os.path.join(MODELS_DIR, "pts_in_hull.npy")

PROTO_URL = "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt"
MODEL_URL = "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel"
HULL_URL = "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy"

# Inject beautiful CSS for a premium glassmorphic dark theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Title Header Style */
.header-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    text-align: center;
}

.header-title {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.05rem;
}

.header-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Feature card styles */
.feature-card {
    background-color: rgba(30, 41, 59, 0.4);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Sidebar Styling */
div[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Beautiful custom buttons */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
    color: white !important;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    width: 100%;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
    background: linear-gradient(90deg, #4338ca 0%, #2563eb 100%);
}

/* Subheaders */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.stTabs [data-baseweb="tab"] {
    height: 3rem;
    font-weight: 600;
    color: #94a3b8;
    background-color: transparent;
    transition: color 0.3s ease;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
}

</style>
""", unsafe_allow_html=True)

# ----------------- Downloader Helper -----------------
def download_model_file(url, dest_path, filename):
    """Downloads a file with a Streamlit progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return True

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.text(f"Initializing download: {filename}")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            bytes_so_far = 0
            block_size = 1024 * 64  # 64KB chunks
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_so_far += len(chunk)
                    if total_size > 0:
                        percent = min(bytes_so_far / total_size, 1.0)
                        progress_bar.progress(percent)
                        status_text.text(f"Downloading {filename}: {percent*100:.1f}% ({bytes_so_far/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)")
            
        status_text.success(f"Successfully downloaded {filename}!")
        progress_bar.empty()
        return True
    except Exception as e:
        status_text.error(f"Error downloading {filename}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

# ----------------- Header Layout -----------------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">CHROMATIC AI</h1>
    <p class="header-subtitle">Deep Learning Grayscale-to-Color Engine with OpenCV & Zhang et al. Architecture</p>
</div>
""", unsafe_allow_html=True)

# Check model status
models_ready = (
    os.path.exists(PROTO_PATH) and 
    os.path.exists(MODEL_PATH) and 
    os.path.exists(HULL_PATH)
)

if not models_ready:
    st.info("👋 Welcome! To start colorizing, we need to download the pre-trained Caffe models (~129MB).")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Download AI Models"):
            success = True
            success &= download_model_file(PROTO_URL, PROTO_PATH, "colorization_deploy_v2.prototxt")
            success &= download_model_file(HULL_URL, HULL_PATH, "pts_in_hull.npy")
            success &= download_model_file(MODEL_URL, MODEL_PATH, "colorization_release_v2.caffemodel")
            
            if success:
                st.success("All models downloaded successfully! Reloading application...")
                st.rerun()
            else:
                st.error("Failed downloading some model components. Please check your internet connection and try again.")
    st.stop()

# ----------------- Load Colorizer -----------------
@st.cache_resource
def get_cached_colorizer(proto, model, hull):
    colorizer = ImageColorizer(proto, model, hull)
    colorizer.load_model()
    return colorizer

with st.spinner("Loading deep learning model weights..."):
    try:
        colorizer = get_cached_colorizer(PROTO_PATH, MODEL_PATH, HULL_PATH)
    except Exception as e:
        st.error(f"Error loading colorization neural network: {e}")
        st.stop()

# ----------------- Sidebar Navigation -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>🔧 Control Dashboard</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Select Application Mode", ["📸 Single Image Colorizer", "📂 Batch Processing"])

# Common post-processing settings for fine-tuning colors
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Color & Tone Adjustments")

saturation = st.sidebar.slider("Color Saturation", 0.0, 2.5, 1.0, 0.1, help="Multiply color vibrancy. 0.0 = Grayscale.")
temp_a = st.sidebar.slider("Color Tint (Green-Red)", -50.0, 50.0, 0.0, 1.0, help="Shift towards red (positive) or green (negative).")
temp_b = st.sidebar.slider("Color Temperature (Blue-Yellow)", -50.0, 50.0, 0.0, 1.0, help="Shift towards yellow (warm, positive) or blue (cool, negative).")
brightness = st.sidebar.slider("Luminance (Brightness)", -30.0, 30.0, 0.0, 1.0, help="Adjust Lightness offset.")
contrast = st.sidebar.slider("Luminance (Contrast)", 0.5, 2.0, 1.0, 0.1, help="Scale light intensity variations.")
sharpen = st.sidebar.slider("Detail Sharpening", 0.0, 1.0, 0.0, 0.1, help="Blend image detail sharpening filter.")

if st.sidebar.button("🔄 Reset Parameters"):
    st.rerun()

# ----------------- App Logic -----------------
if app_mode == "📸 Single Image Colorizer":
    st.markdown("### 📸 Colorize Single Image")
    
    # Check if samples exist, if so populate them
    sample_files = []
    if os.path.exists(SAMPLES_DIR):
        sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # File input
    source_choice = st.radio("Choose Input Method", ["Upload your own photo", "Select from Sample Gallery"], horizontal=True)
    img_bgr = None
    
    if source_choice == "Upload your own photo":
        uploaded_file = st.file_uploader("Upload a Grayscale or Color image (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        if sample_files:
            selected_sample = st.selectbox("Select a Sample Photo", sample_files)
            if selected_sample:
                sample_path = os.path.join(SAMPLES_DIR, selected_sample)
                img_bgr = cv2.imread(sample_path)
        else:
            st.warning("No sample gallery files found. Run the sample generator first or upload your own file.")

    if img_bgr is not None:
        # Step 1: Ensure image is converted to grayscale first to demonstrate baseline B&W colorization
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Step 2: Run colorization
        with st.spinner("Processing through deep learning colorizer..."):
            try:
                colorized_bgr = colorizer.colorize(
                    img_gray_bgr,
                    saturation=saturation,
                    temp_a=temp_a,
                    temp_b=temp_b,
                    brightness=brightness,
                    contrast=contrast,
                    sharpen=sharpen
                )
            except Exception as e:
                st.error(f"Inference Error: {e}")
                colorized_bgr = None
        
        if colorized_bgr is not None:
            # Create tabs for Output vs Data Analysis
            tab1, tab2 = st.tabs(["🎨 Colorization Result", "📊 Data Analysis & Histograms"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='feature-card'><h4>Grayscale Input</h4></div>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(img_gray_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.markdown("<div class='feature-card'><h4>Chromatic AI Colorized</h4></div>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Download Button
                is_success, buffer = cv2.imencode(".png", colorized_bgr)
                if is_success:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="⬇️ Download Colorized Image (PNG)",
                        data=buffer.tobytes(),
                        file_name="colorized_output.png",
                        mime="image/png"
                    )
            
            with tab2:
                st.markdown("<div class='feature-card'><h4>Intensity & Color Distribution Comparison</h4></div>", unsafe_allow_html=True)
                with st.spinner("Generating statistical histograms..."):
                    plot_buf = generate_histogram_plot(img_gray_bgr, colorized_bgr)
                    st.image(plot_buf, use_container_width=True)
                    st.markdown("""
                    **Histogram Insights:**
                    - **Grayscale Input Distribution:** Shows the density of brightness levels (lightness/luminosity).
                    - **Colorized Output Distributions:** Illustrates how the Caffe convolutional network split the grayscale energy across the Red, Green, and Blue spectrums to reconstruct a balanced and plausible color gamut.
                    """)

else:
    # ----------------- Batch Processing Mode -----------------
    st.markdown("### 📂 Batch Process Grayscale Photos")
    st.markdown("Upload multiple images to colorize them in a single batch with the active slider configuration. When completed, you can download all results in a single structured ZIP file.")
    
    uploaded_files = st.file_uploader("Upload Grayscale/Color Photos", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🚀 Process Batch"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            zip_buffer = io.BytesIO()
            total_files = len(uploaded_files)
            
            # Temporary storage to display grid of results
            results_to_show = []
            
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Colorizing file {idx+1}/{total_files}: {uploaded_file.name}")
                    
                    # Read image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if img_bgr is not None:
                        # Convert to grayscale first
                        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        img_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        
                        # Process image
                        colorized_bgr = colorizer.colorize(
                            img_gray_bgr,
                            saturation=saturation,
                            temp_a=temp_a,
                            temp_b=temp_b,
                            brightness=brightness,
                            contrast=contrast,
                            sharpen=sharpen
                        )
                        
                        # Save inside zip
                        img_name, img_ext = os.path.splitext(uploaded_file.name)
                        out_filename = f"{img_name}_colorized{img_ext}"
                        
                        # Encode back to original extension
                        is_success, encoded_img = cv2.imencode(img_ext, colorized_bgr)
                        if is_success:
                            zip_file.writestr(out_filename, encoded_img.tobytes())
                            
                            # Keep record for visual review (limit to first 10 for performance)
                            if idx < 10:
                                results_to_show.append((uploaded_file.name, img_gray_bgr, colorized_bgr))
                                
                    progress_bar.progress((idx + 1) / total_files)
            
            status_text.success(f"Successfully processed {total_files} files!")
            progress_bar.empty()
            
            # Offer download
            zip_buffer.seek(0)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label=f"⬇️ Download Colorized ZIP ({total_files} images)",
                data=zip_buffer.getvalue(),
                file_name="colorized_images_batch.zip",
                mime="application/zip"
            )
            
            # Show preview grid
            if results_to_show:
                st.markdown("---")
                st.markdown("### 🔍 Batch Preview (First 10 Images)")
                for name, gray_img, col_img in results_to_show:
                    with st.expander(f"Preview: {name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB), caption="Grayscale Input", use_container_width=True)
                        with col2:
                            st.image(cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB), caption="Colorized Output", use_container_width=True)
