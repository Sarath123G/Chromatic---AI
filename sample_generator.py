import os
import cv2
import urllib.request
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Standard B&W/Grayscale or easily colorizable images from official OpenCV repository
SAMPLE_URLS = {
    "Lena (Color -> B&W conversion)": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "Building (Grayscale)": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    "Butterfly (Color -> B&W conversion)": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg",
    "Fruits (Color -> B&W conversion)": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg"
}

def generate_samples():
    print("Setting up sample gallery images...")
    for name, url in SAMPLE_URLS.items():
        filename = url.split("/")[-1]
        dest_path = os.path.join(SAMPLES_DIR, filename)
        
        # Determine name to save
        if "conversion" in name:
            base, ext = os.path.splitext(filename)
            dest_path = os.path.join(SAMPLES_DIR, f"{base}_bw{ext}")
            
        if os.path.exists(dest_path):
            print(f"Sample '{dest_path}' already exists. Skipping.")
            continue
            
        try:
            print(f"Downloading {name} from {url}...")
            # Download file into memory
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                img_data = response.read()
                
            # Decode to cv2 image
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Failed to decode image from {url}")
                continue
                
            # If it's a color conversion sample, convert to grayscale
            if "conversion" in name:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Save as grayscale
                cv2.imwrite(dest_path, gray)
                print(f"Saved color image converted to grayscale: {dest_path}")
            else:
                # Save grayscale as is
                cv2.imwrite(dest_path, img)
                print(f"Saved grayscale image: {dest_path}")
                
        except Exception as e:
            print(f"Failed to download/process {name}: {e}")

if __name__ == "__main__":
    generate_samples()
