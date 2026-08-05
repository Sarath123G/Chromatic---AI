import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

class ImageColorizer:
    def __init__(self, proto_path, model_path, hull_path):
        self.proto_path = proto_path
        self.model_path = model_path
        self.hull_path = hull_path
        self.net = None
        self.is_loaded = False

    def load_model(self):
        """Loads the Caffe colorization model and setups custom cluster center convolutions."""
        if self.is_loaded:
            return True
        
        if not os.path.exists(self.proto_path) or not os.path.exists(self.model_path) or not os.path.exists(self.hull_path):
            raise FileNotFoundError("Model files not found. Please ensure they are downloaded first.")
            
        try:
            # Read network structure and weights
            # Use generic readNet which works in OpenCV 4.x AND 5.x
            # (readNetFromCaffe was removed in OpenCV 5.0)
            if hasattr(cv2.dnn, 'readNetFromCaffe'):
                self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
            else:
                self.net = cv2.dnn.readNet(self.proto_path, self.model_path)
            
            # Load cluster points (pts_in_hull.npy)
            kernel = np.load(self.hull_path)
            
            # Add cluster centers as 1x1 convolutions to the network
            class8 = self.net.getLayerId("class8_ab")
            conv8 = self.net.getLayerId("conv8_313_rh")
            
            # Reshape cluster points
            pts = kernel.transpose().reshape(2, 313, 1, 1)
            
            # Set model blobs
            self.net.getLayer(class8).blobs = [pts.astype("float32")]
            self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            
            self.is_loaded = True
            return True
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"Error loading model: {e}")

    def colorize(self, img_bgr, saturation=1.0, temp_a=0.0, temp_b=0.0, brightness=0.0, contrast=1.0, sharpen=0.0):
        """
        Runs colorization on the input BGR image.
        
        Parameters:
        - img_bgr: Input BGR image (numpy array)
        - saturation: Multiplier for the ab color channels (0.0 to 2.5)
        - temp_a: Bias added to the 'a' channel (green-red) to alter color temperature
        - temp_b: Bias added to the 'b' channel (blue-yellow) to alter color temperature
        - brightness: Offset added to the L channel
        - contrast: Multiplier for the L channel scaling
        - sharpen: Amount of sharpening to blend in (0.0 to 1.0)
        
        Returns:
        - colorized_bgr: The colorized BGR image
        """
        if not self.is_loaded:
            self.load_model()
            
        # Ensure image is in BGR format
        h, w = img_bgr.shape[:2]
        
        # Preprocessing: Scale image to [0, 1] and convert to Lab color space
        scaled = img_bgr.astype("float32") / 255.0
        lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        
        # Resize Lab image to 224x224 (required input shape for the model)
        resized = cv2.resize(lab_img, (224, 224))
        
        # Split the channels and extract the 'L' (lightness) channel
        L_resized = cv2.split(resized)[0]
        # Mean subtraction (standard training offset)
        L_resized -= 50
        
        # Set input to network — blobFromImage packs L into a 4D blob (1,1,224,224)
        blob = cv2.dnn.blobFromImage(L_resized)
        self.net.setInput(blob)
        
        # Predict ab channels (returns shape [1, 2, 56, 56])
        ab_predicted = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        # Resize predicted ab volume back to the original image dimensions
        ab_predicted = cv2.resize(ab_predicted, (w, h))
        
        # Extract the original L channel
        L_orig = cv2.split(lab_img)[0]
        
        # --- Post-Processing 1: Brightness & Contrast on Lightness channel ---
        if brightness != 0.0 or contrast != 1.0:
            # Shift lightness around the mid-tone (50) for contrast
            L_orig = (L_orig - 50.0) * contrast + 50.0 + brightness
            L_orig = np.clip(L_orig, 0.0, 100.0)
            
        # --- Post-Processing 2: Saturation scaling on predicted color channels ---
        if saturation != 1.0:
            ab_predicted = ab_predicted * saturation
            
        # --- Post-Processing 3: Color temperature shifts on ab channels ---
        if temp_a != 0.0:
            ab_predicted[:, :, 0] += temp_a
        if temp_b != 0.0:
            ab_predicted[:, :, 1] += temp_b
            
        ab_predicted = np.clip(ab_predicted, -128.0, 127.0)
        
        # Rejoin L and ab channels
        colorized = np.concatenate((L_orig[:, :, np.newaxis], ab_predicted), axis=2)
        
        # Convert back from Lab to BGR color space
        colorized_bgr = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized_bgr = np.clip(colorized_bgr, 0.0, 1.0)
        colorized_bgr = (255.0 * colorized_bgr).astype("uint8")
        
        # --- Post-Processing 4: Detail Sharpening ---
        if sharpen > 0.0:
            # Sharpening kernel
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1,  9, -1],
                                          [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(colorized_bgr, -1, kernel_sharpening)
            colorized_bgr = cv2.addWeighted(colorized_bgr, 1.0 - sharpen, sharpened, sharpen, 0)
            
        return colorized_bgr

def generate_histogram_plot(img_bgr, colorized_bgr):
    """Generates a comparison matplotlib figure as an in-memory buffer."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # 1. Grayscale histogram
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    axes[0].hist(gray.ravel(), 256, [0, 256], color='#555555', alpha=0.7)
    axes[0].set_title('Original Grayscale Pixel Intensity', fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlabel('Lightness Value', fontsize=9)
    axes[0].set_ylabel('Count', fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # 2. Colorized RGB histogram
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
    colors = ('#E74C3C', '#2ECC71', '#3498DB') # Darker Red, Green, Blue
    labels = ('Red Channel', 'Green Channel', 'Blue Channel')
    
    for i, (col, label) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([colorized_rgb], [i], None, [256], [0, 256])
        axes[1].plot(hist, color=col, label=label, alpha=0.8, linewidth=2.0)
        
    axes[1].set_title('Colorized Image RGB Channel Distributions', fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xlabel('Channel Value', fontsize=9)
    axes[1].set_ylabel('Count', fontsize=9)
    axes[1].legend(frameon=True, facecolor='#ffffff', edgecolor='none')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf
