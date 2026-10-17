import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import csv
from openvino.runtime import Core
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from collections import deque

rtsp_url="rtsp://10.9.174.23:8080/h264.sdp"

def load_model_and_encodings(model_path):
    """Loads the trained model and face encodings."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_faces_batch(compiled_model, input_shape, encodings, names, faces):
    """Recognizes multiple faces in parallel."""
    h, w = input_shape[2], input_shape[3]
    results = []

    for face_img, face_coords in faces:
        resized_face = cv2.resize(face_img, (w, h))
        input_data = np.expand_dims(resized_face.transpose(2, 0, 1), axis=0)

        result = compiled_model([input_data])[compiled_model.output(0)]
        face_encoding = result.flatten()

        # Match with known encodings
        similarities = cosine_similarity([face_encoding], encodings)
        max_similarity_idx = np.argmax(similarities)

        if similarities[0, max_similarity_idx] > 0.5:  # Threshold for recognition
            results.append((names[max_similarity_idx], face_coords))

    return results

def detect_faces(frame, face_cascade):
    """Detects faces in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(frame[y:y+h, x:x+w], (x, y, w, h)) for (x, y, w, h) in faces]

def log_to_csv(log_path, name, logged_names):
    """Logs the recognized face with timestamp to a CSV file."""
    if name not in logged_names:
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        logged_names.add(name)

def main():
    # Use relative paths based on script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "model.pkl")
    openvino_model_path = os.path.join(base_dir, "converted_ir", "new", "arcfaceresnet100-8.xml")
    log_file = os.path.join(base_dir, "attendance", "attendance.csv")

    # Load the trained model and encodings
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    known_encodings, known_names = load_model_and_encodings(model_path)

    # Load OpenVINO model
    if not os.path.exists(openvino_model_path):
        print(f"Error: OpenVINO model not found at {openvino_model_path}")
        return

    core = Core()
    model = core.read_model(model=openvino_model_path)
    compiled_model = core.compile_model(model, device_name="CPU")
    input_shape = compiled_model.input(0).shape

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit.")

    logged_names = set()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            faces = detect_faces(frame, face_cascade)
            if faces:
                future = executor.submit(recognize_faces_batch, compiled_model, input_shape, known_encodings, known_names, faces)
                recognized_faces = future.result()

                for name, (x, y, w, h) in recognized_faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    log_to_csv(log_file, name, logged_names)

            cv2.imshow("Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
