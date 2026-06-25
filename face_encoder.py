import face_recognition, os, pickle

def encode_faces(folder="known_faces"):
    encodings, names = [], []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg",".jpeg",".png")):
            img = face_recognition.load_image_file(f"{folder}/{f}")
            enc_list = face_recognition.face_encodings(img)
            if enc_list:
                encodings.append(enc_list[0])
                names.append(os.path.splitext(f)[0])
    with open("face_data.pkl","wb") as fp:
        pickle.dump({"encodings":encodings,"names":names}, fp)
    print(f"Encoded {len(names)} faces: {names}")

if __name__ == "__main__":
    encode_faces()