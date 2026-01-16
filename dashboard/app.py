import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Object Detection Dashboard", layout="wide")

API_URL = st.sidebar.text_input(
    "API URL",
    value="https://object-detection-api-rmtj.onrender.com"
)

page = st.sidebar.radio(
    "View",
    ["Detect", "Statistics", "History"]
)

if page == "Detect":
    st.title("Object Detection")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    confidence = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05)

    if uploaded and st.button("Run detection"):
        response = requests.post(
            f"{API_URL}/detect",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            params={"confidence": confidence},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            st.success(f"Detected {data['total_objects']} objects")
            st.json(data)

            img_response = requests.post(
                f"{API_URL}/detect/annotated",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                params={"confidence": confidence},
                timeout=30
            )

            if img_response.status_code == 200:
                image = Image.open(io.BytesIO(img_response.content))
                st.image(image, use_container_width=True)
        else:
            st.error(response.text)

elif page == "Statistics":
    st.title("Statistics")
    response = requests.get(f"{API_URL}/stats", timeout=10)
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error(response.text)

elif page == "History":
    st.title("Detection History")
    response = requests.get(f"{API_URL}/history", timeout=10)
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error(response.text)