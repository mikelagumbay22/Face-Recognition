import streamlit as st, face_recognition, numpy as np, pickle
from PIL import Image

st.title("Face Recognition")

if st.button("Encode known faces from folder"):
    from face_encoder import encode_faces
    encode_faces()
    st.success("Faces encoded and saved!")

uploaded = st.file_uploader("Upload photo to identify", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)
    arr = np.array(img)
    try:
        with open("face_data.pkl","rb") as fp:
            data = pickle.load(fp)
        locs = face_recognition.face_locations(arr)
        encs = face_recognition.face_encodings(arr, locs)
        results = []
        for enc in encs:
            matches = face_recognition.compare_faces(data["encodings"],enc,tolerance=0.5)
            name = data["names"][matches.index(True)] if True in matches else "Unknown"
            results.append(name)
        st.write(f"Found: {', '.join(results) if results else 'No faces detected'}")
    except FileNotFoundError:
        st.error("No face database found. Click Encode button first.")