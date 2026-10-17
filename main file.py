import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import csv
from openvino.runtime import Core
from sklearn.metrics.pairwise import cosine_similarity


rtsp_url="rtsp://192.168.1.10:8080/h264.sdp"
def load_model_and_encodings(model_path):
    """Loads the trained model and face encodings."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_face(compiled_model, input_shape, encodings, names, frame):
    """Recognizes faces in the given frame."""
    h, w = input_shape[2], input_shape[3]

    # Convert frame to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    recognized_faces = []

    for (x, y, face_w, face_h) in faces:
        face_img = frame[y:y+face_h, x:x+face_w]
        resized_face = cv2.resize(face_img, (w, h))
        input_data = np.expand_dims(resized_face.transpose(2, 0, 1), axis=0)

        result = compiled_model([input_data])[compiled_model.output(0)]
        face_encoding = result.flatten()

        # Match with known encodings
        similarities = cosine_similarity([face_encoding], encodings)
        max_similarity_idx = np.argmax(similarities)

        if similarities[0, max_similarity_idx] > 0.6:  # Threshold for recognition
            recognized_faces.append((names[max_similarity_idx], (x, y, face_w, face_h)))

    return recognized_faces

def log_to_csv(log_path, name):
    """Logs the recognized face with timestamp to a CSV file."""
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

if __name__ == "__main__":
    model_path =r"D:\Project\model\model.pkl"
    openvino_model_path = r"D:\Project\converted_ir\new\arcfaceresnet100-8.xml"
    log_file = r"D:\Project\attendance.csv"

    # Load the trained model and encodings
    known_encodings, known_names = load_model_and_encodings(model_path)

    # Load OpenVINO model
    core = Core()
    model = core.read_model(model=openvino_model_path)
    compiled_model = core.compile_model(model, device_name="GPU")
    input_shape = compiled_model.input(0).shape

    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        recognized_faces = recognize_face(compiled_model, input_shape, known_encodings, known_names, frame)

        for name, (x, y, face_w, face_h) in recognized_faces:
            cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            log_to_csv(log_file, name)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
