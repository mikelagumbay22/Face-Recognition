import face_recognition, cv2, pickle

def run():
    with open("face_data.pkl","rb") as fp:
        data = pickle.load(fp)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        for enc,(top,right,bottom,left) in zip(encs,locs):
            matches = face_recognition.compare_faces(data["encodings"],enc,tolerance=0.5)
            name = data["names"][matches.index(True)] if True in matches else "Unknown"
            t,r,b,l = top*4,right*4,bottom*4,left*4
            cv2.rectangle(frame,(l,t),(r,b),(0,200,100),2)
            cv2.putText(frame,name,(l,t-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,100),2)
        cv2.imshow("Face Recognition - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()