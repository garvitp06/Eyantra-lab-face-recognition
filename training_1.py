import os
import cv2
import numpy as np
import pickle
from openvino.runtime import Core
from sklearn.preprocessing import normalize
rtsp_url="rtsp://10.9.174.23:8080/h264.sdp"
def capture_faces(output_dir, person_name, num_images=100):
    """Captures faces for a specific person and saves them to the output directory."""
    person_dir = os.path.join(output_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    cap = cv2.VideoCapture(0)
    print(f"Capturing images for {person_name}. Press 'q' to quit early.")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            output_path = os.path.join(person_dir, f"face_{count}.jpg")
            cv2.imwrite(output_path, face_img)
            print(f"Captured {output_path}")
            count += 1

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def encode_faces_with_openvino(dataset_dir, model_path):
    """Encodes faces using OpenVINO and saves the model."""
    core = Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device_name="CPU")
    input_layer = compiled_model.input(0)

    input_shape = input_layer.shape  # Get the expected input shape
    n, c, h, w = input_shape  # Typically [1, 3, height, width]

    known_encodings = []
    known_names = []

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            resized_image = cv2.resize(image, (w, h))  # Resize to the expected input size
            input_data = np.expand_dims(resized_image.transpose(2, 0, 1), axis=0)  # Prepare input

            result = compiled_model([input_data])[compiled_model.output(0)]
            encoding = result.flatten()
            normalized_encoding = normalize([encoding])[0]

            known_encodings.append(normalized_encoding)
            known_names.append(person_name)

    return known_encodings, known_names

def train_and_save_model(dataset_dir, model_path, output_model_path):
    """Trains the model using the dataset and saves it to disk."""
    print("Encoding faces with OpenVINO...")
    known_encodings, known_names = encode_faces_with_openvino(dataset_dir, model_path)

    print("Saving model...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(output_model_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    dataset_dir = input("Enter the path to save the dataset directory: ").strip()
    model_path = "D:/Project/converted_ir/new/arcfaceresnet100-8.xml"  # Path to your OpenVINO model XML file
    output_model_path = input("Enter the path to save the trained model (e.g., face_recognition_model.pkl): ").strip()

    # Step 1: Capture faces for multiple people.
    while True:
        person_name = input("Enter the name of the person (or type 'done' to finish): ").strip()
        if person_name.lower() == 'done':
            break
        capture_faces(dataset_dir, person_name, num_images=100)

    # Step 2: Train the model and save it.
    train_and_save_model(dataset_dir, model_path, output_model_path)
