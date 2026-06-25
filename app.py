import os
import pickle

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

KNOWN_FACES_DIR = "known_faces"
FACE_DATA_FILE = "face_data.pkl"


@st.cache_resource
def load_face_recognition():
    try:
        import face_recognition

        return face_recognition, None
    except ImportError as exc:
        return None, str(exc)


def find_known_face_image(name):
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(KNOWN_FACES_DIR, f"{name}{ext}")
        if os.path.isfile(path):
            return path
    return None


@st.cache_data
def load_face_database():
    face_recognition, import_error = load_face_recognition()
    if import_error:
        raise RuntimeError(import_error)

    if os.path.isfile(FACE_DATA_FILE):
        with open(FACE_DATA_FILE, "rb") as fp:
            return pickle.load(fp)

    from face_encoder import encode_faces

    encode_faces()
    with open(FACE_DATA_FILE, "rb") as fp:
        return pickle.load(fp)


def identify_faces(image_array, data, face_recognition):
    rgb = image_array[:, :, :3]
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    results = []

    for enc, (top, right, bottom, left) in zip(encs, locs):
        matches = face_recognition.compare_faces(data["encodings"], enc, tolerance=0.5)
        name = data["names"][matches.index(True)] if True in matches else "Unknown"
        results.append(
            {
                "name": name,
                "box": (left, top, right, bottom),
                "known_image": find_known_face_image(name) if name != "Unknown" else None,
            }
        )

    return results


def draw_labels(image, results):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for result in results:
        left, top, right, bottom = result["box"]
        draw.rectangle((left, top, right, bottom), outline=(0, 200, 100), width=3)
        draw.text((left, max(top - 18, 0)), result["name"], fill=(0, 200, 100))
    return annotated


st.set_page_config(page_title="Face Recognition", page_icon="📷", layout="wide")
st.title("Face Recognition")
st.caption("Use your camera to scan a face and match it against known people.")

if not os.path.isdir(KNOWN_FACES_DIR):
    st.warning(f"`{KNOWN_FACES_DIR}/` folder not found. Add reference photos and redeploy.")
    st.stop()

known_images = [
    f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
if not known_images:
    st.warning(f"No photos found in `{KNOWN_FACES_DIR}/`. Add `.jpg` or `.png` files and redeploy.")
    st.stop()

with st.spinner("Loading face recognition (first run can take a few seconds)..."):
    face_recognition, import_error = load_face_recognition()
    if import_error:
        st.error("Could not load `face_recognition`.")
        st.code(import_error)
        st.info("Run locally: `pip install -r requirements.txt` then restart the app.")
        st.stop()

    try:
        data = load_face_database()
    except Exception as exc:
        st.error("Could not build the face database from known_faces.")
        st.code(str(exc))
        st.stop()

st.success(f"Loaded {len(data['names'])} known face(s): {', '.join(data['names'])}")

with st.expander("Known people"):
    cols = st.columns(min(len(data["names"]), 4))
    for idx, name in enumerate(data["names"]):
        image_path = find_known_face_image(name)
        with cols[idx % len(cols)]:
            if image_path:
                st.image(image_path, caption=name, use_container_width=True)
            else:
                st.write(name)

st.subheader("Scan with camera")
camera_photo = st.camera_input("Look at the camera, then capture a photo")

uploaded = st.file_uploader("Or upload a photo instead", type=["jpg", "jpeg", "png"])

image_source = camera_photo or uploaded
if image_source is not None:
    img = Image.open(image_source).convert("RGB")
    arr = np.array(img)
    results = identify_faces(arr, data, face_recognition)

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Captured photo")
        if results:
            st.image(draw_labels(img, results), use_container_width=True)
        else:
            st.image(img, use_container_width=True)
            st.warning("No face detected. Try better lighting and face the camera directly.")

    with right_col:
        st.subheader("Match result")
        if not results:
            st.info("Capture a clear photo of a face to identify.")
        else:
            for result in results:
                st.write(f"**{result['name']}**")
                if result["known_image"]:
                    st.image(result["known_image"], caption=f"Known photo: {result['name']}")
                elif result["name"] == "Unknown":
                    st.info("No match found in known_faces.")
