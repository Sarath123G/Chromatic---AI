# Chromatic AI - Deep Image Colorizer

Chromatic AI is a powerful web application that breathes life into grayscale images using deep learning. It leverages the Zhang et al. colorization architecture via OpenCV's Deep Neural Network (DNN) module and presents a beautiful, interactive interface built with Streamlit.

## Features

*   **Deep Learning Colorization:** Automatically colorizes black and white images using a pre-trained Caffe model.
*   **Interactive Adjustments:** Fine-tune the colorized output with real-time sliders for:
    *   Color Saturation
    *   Color Tint (Green-Red)
    *   Color Temperature (Blue-Yellow)
    *   Luminance (Brightness & Contrast)
    *   Detail Sharpening
*   **Data Analysis:** View side-by-side histograms comparing original grayscale pixel intensity with the predicted RGB channel distributions.
*   **Batch Processing:** Upload multiple images, process them simultaneously with your chosen slider settings, and download all results in a single ZIP file.
*   **Sample Gallery:** Quickly test the app using provided historical sample photos.

## Setup Instructions for Local Development

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd "Chromatic AI"
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

4.  **Download Models (First Run):**
    When you open the app for the first time, click the **"🚀 Download AI Models"** button. This will securely download the required ~129MB Caffe model files into the `models/` directory.

## Deployment to Streamlit Community Cloud

This repository is fully configured for deployment on Streamlit Community Cloud.

1.  Push this repository to your GitHub account.
2.  Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3.  Click **"New app"**.
4.  Select your GitHub repository, set the branch to `main`, and the Main file path to `app.py`.
5.  Click **"Deploy!"**

*Note: The large `.caffemodel` file is managed via Git LFS. Streamlit Community Cloud supports Git LFS automatically.*

## Technologies Used

*   Python
*   Streamlit (UI Framework)
*   OpenCV (Image Processing & DNN inference)
*   NumPy (Array manipulation)
*   Matplotlib (Histogram generation)
