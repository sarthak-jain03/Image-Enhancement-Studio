import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Enhancement Studio", layout="wide")

st.title("Image Enhancement Studio")
st.write("Explore Digital Image Processing concepts: intensity transformation, histogram equalization, and CLAHE.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def adjust_brightness_contrast(image, brightness=0, contrast=0):
   
    img = np.int16(image)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def plot_histogram(image, title):
    color = ('b', 'g', 'r')
    fig, ax = plt.subplots()
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   
    st.sidebar.header("Adjust Parameters 🧭")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", -100, 100, 0)
    gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0)

    apply_equalization = st.sidebar.checkbox("Apply Histogram Equalization (Global)")
    apply_clahe = st.sidebar.checkbox("Apply CLAHE (Local Equalization)")

   
    enhanced = adjust_brightness_contrast(img, brightness, contrast)
    enhanced = gamma_correction(enhanced, gamma)

   
    if apply_equalization:
        img_yuv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    elif apply_clahe:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
        plot_histogram(img, "Original Image Histogram")
    with col2:
        st.image(enhanced, caption="Enhanced Image", use_container_width=True)
        plot_histogram(enhanced, "Enhanced Image Histogram")

else:
    st.info("Please upload an image to begin.")
