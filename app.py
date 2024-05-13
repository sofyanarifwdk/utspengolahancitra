import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

def rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image

def calculate_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])
    return hist

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131 - contrast))
        alpha_c = f
        gamma_c = 127*(1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    st.title("Web App untuk Manipulasi Gambar")
    uploaded_file = st.file_uploader("Unggah gambar", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # st.subheader("Gambar Asli")
        st.image(image, caption='Gambar Asli', use_column_width=True)

        hsv_image = rgb_to_hsv(image)
        st.image(hsv_image, caption='Gambar HSV', use_column_width=True)

        histogram = calculate_histogram(image)
        st.write("Histogram Gambar:")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(histogram)
        ax.set_title("Histogram Gambar")
        ax.set_xlabel("Intensitas")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
        
        brightness = st.slider("Kecerahan", -100, 100, 0)
        contrast = st.slider("Kontras", -100, 100, 0)
        adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
        st.image(adjusted_image, caption='Gambar Dengan Penyesuaian', use_column_width=True)
        
        contours = find_contours(cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR)) # Convert to BGR before finding contours
        image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
        st.image(image_with_contours, caption='Gambar dengan Kontur', use_column_width=True)

if __name__ == "__main__":
    main()
