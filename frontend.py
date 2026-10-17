import sys
import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import csv
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from openvino.runtime import Core
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from main_file_1 import detect_faces, recognize_faces_batch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
                             QProgressBar, QComboBox, QFrame, QSplitter, QDialog, QFormLayout,
                             QCalendarWidget, QDateEdit)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QDate
import os

# Project base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(rel_path):
    """Convert relative path to absolute path based on project root."""
    if not rel_path:
        return ""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(BASE_DIR, rel_path)


class FaceRecognitionThread(QThread):
    update_frame = pyqtSignal(np.ndarray, list)

    def __init__(self, parent=None, openvino_model_path=None, model_path=None):
        super().__init__(parent)
        self.running = False
        self.openvino_model_path = openvino_model_path
        self.model_path = model_path
        self.camera_source = 0

    def set_camera_source(self, source):
        self.camera_source = source

    def load_model(self, model_path, openvino_model_path):
        self.model_path = model_path
        self.openvino_model_path = openvino_model_path

        # Load the trained model and encodings
        with open(get_path(self.model_path), "rb") as f:
            data = pickle.load(f)
        self.known_encodings = data["encodings"]
        self.known_names = data["names"]

        # Load OpenVINO model
        self.core = Core()
        self.model = self.core.read_model(model=get_path(self.openvino_model_path))
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU")
        self.input_shape = self.compiled_model.input(0).shape

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def show_camera_error(self, message):
        QMessageBox.critical(self, "Camera Error",
                             f"{message}\n\n"
                             "Please check:\n"
                             "1. Camera is properly connected\n"
                             "2. Correct camera source is selected\n"
                             "3. Camera permissions are granted")
    def run(self):
        self.logged_names = set()
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap.isOpened():
            self.parent().show_camera_error("Failed to open video source")
            self.running = False
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Use detect_faces from main_file_1
            faces = detect_faces(frame, self.face_cascade)
            
            recognized_faces = []
            if faces:
                # Use recognize_faces_batch from main_file_1
                # recognize_faces_batch returns [(name, coords), ...]
                # We need to adapt it to (name, coords, confidence) if needed, 
                # but main_file_1 version doesn't return confidence. 
                # I'll modify the thread to handle the returned structure.
                batch_results = recognize_faces_batch(
                    self.compiled_model, 
                    self.input_shape, 
                    self.known_encodings, 
                    self.known_names, 
                    faces
                )
                
                for name, coords in batch_results:
                    # main_file_1 doesn't calculate confidence in the batch function, 
                    # so we'll pass 1.0 or similar.
                    recognized_faces.append((name, coords, 1.0))

            self.update_frame.emit(frame, recognized_faces)

        if self.cap is not None:
            self.cap.release()

    # Modify the stop methods to ensure proper release
    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.wait()

class FaceCapturingThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    update_progress = pyqtSignal(int)
    capture_complete = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.capture_active = False # Initialized here to avoid AttributeError
        self.count = 0
        self.person_name = None
        self.dataset_dir = None
        self.num_images = 50
        self.camera_source = 0

    def set_camera_source(self, source):
        self.camera_source = source

    def set_capture_params(self, dataset_dir, person_name, num_images):
        self.dataset_dir = dataset_dir
        self.person_name = person_name
        self.num_images = num_images
        self.capture_active = False 

    def start_capture(self):
        self.capture_active = True
        self.count = 0

    def stop_capture(self):
        self.capture_active = False

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_source)
        self.count = 0
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if self.capture_active and self.count < self.num_images:
                # Visual Feedback / Animation: Draw a green border when capturing
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
                cv2.putText(frame, f"CAPTURING: {self.count}/{self.num_images}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    # Use absolute path for saving
                    abs_dataset_dir = get_path(self.dataset_dir)
                    person_dir = os.path.join(abs_dataset_dir, self.person_name)
                    if not os.path.exists(person_dir):
                        os.makedirs(person_dir)

                    for (x, y, w, h) in faces:
                        face_img = frame[y:y + h, x:x + w]
                        output_path = os.path.join(person_dir, f"face_{self.count}.jpg")
                        cv2.imwrite(output_path, face_img)
                        self.count += 1
                        self.update_progress.emit(int(self.count * 100 / self.num_images))
                        if self.count >= self.num_images:
                            self.capture_active = False
                            self.capture_complete.emit()
                            break

            self.update_frame.emit(frame)

        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


class ModelTrainingThread(QThread):
    update_progress = pyqtSignal(int, str)
    training_complete = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_dir = None
        self.model_path = None
        self.output_model_path = None

    def set_training_params(self, dataset_dir, model_path, output_model_path):
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.output_model_path = output_model_path

    def run(self):
        self.update_progress.emit(10, "Loading OpenVINO model...")

        abs_model_path = get_path(self.model_path)
        abs_dataset_dir = get_path(self.dataset_dir)
        abs_output_model_path = get_path(self.output_model_path)

        core = Core()
        model = core.read_model(model=abs_model_path)
        compiled_model = core.compile_model(model, device_name="CPU")
        input_layer = compiled_model.input(0)

        input_shape = input_layer.shape
        n, c, h, w = input_shape

        known_encodings = []
        known_names = []

        self.update_progress.emit(20, "Processing dataset...")
        if not os.path.exists(abs_dataset_dir):
            self.update_progress.emit(0, "Error: Dataset directory not found")
            return

        people = [name for name in os.listdir(abs_dataset_dir)
                  if os.path.isdir(os.path.join(abs_dataset_dir, name))]
        person_count = len(people)
        current_person = 0

        for person_name in people:
            person_dir = os.path.join(abs_dataset_dir,  person_name)
            current_person += 1
            self.update_progress.emit(
                20 + int(60 * current_person / person_count),
                f"Processing {person_name}..."
            )

            image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for i, img_name in enumerate(image_files):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                resized_image = cv2.resize(image, (w, h))
                input_data = np.expand_dims(resized_image.transpose(2, 0, 1), axis=0)

                result = compiled_model([input_data])[compiled_model.output(0)]
                encoding = result.flatten()
                normalized_encoding = normalize([encoding])[0]

                known_encodings.append(normalized_encoding)
                known_names.append(person_name)

        self.update_progress.emit(90, "Saving model...")
        data = {"encodings": known_encodings, "names": known_names}
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(abs_output_model_path), exist_ok=True)
        
        with open(abs_output_model_path, "wb") as f:
            pickle.dump(data, f)

        self.update_progress.emit(100, "Training complete!")
        self.training_complete.emit()


class CameraSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Source")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        form_layout = QFormLayout()

        self.source_combo = QComboBox()
        self.source_combo.addItem("Default Camera (0)", 0)
        self.source_combo.addItem("External Camera (1)", 1)
        self.source_combo.addItem("IP Camera/RTSP", "custom")

        form_layout.addRow("Source:", self.source_combo)

        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://username:password@ip_address:port/path")
        self.rtsp_input.setEnabled(False)
        form_layout.addRow("RTSP URL:", self.rtsp_input)

        layout.addLayout(form_layout)

        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

        self.source_combo.currentIndexChanged.connect(self.on_source_changed)

    def on_source_changed(self, index):
        source_data = self.source_combo.currentData()
        self.rtsp_input.setEnabled(source_data == "custom")

    def get_camera_source(self):
        source_data = self.source_combo.currentData()
        if source_data == "custom":
            return self.rtsp_input.text()
        return source_data


class FacialAttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Attendance System")
        self.setMinimumSize(900, 600)

        # Initialize variables with defaults
        self.dataset_dir = "dataset/"
        self.model_path = os.path.join("model", "model.pkl")
        self.openvino_model_path = os.path.join("openvino_model", "arcfaceresnet100-8.xml")
        self.attendance_log_path = os.path.join("attendance", "attendance.csv")
        self.camera_source = 0

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.setup_recognition_tab()
        self.setup_registration_tab()
        self.setup_training_tab()
        self.setup_settings_tab()
        self.setup_attendance_tab()

        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.handle_tab_change)

        # Initialize threads
        self.recognition_thread = None
        self.capture_thread = None
        self.training_thread = None

        # Update UI labels with defaults
        self.dataset_dir_label.setText(self.dataset_dir)
        if hasattr(self, 'registration_dir_label'):
            self.registration_dir_label.setText(f"Dataset Dir: {self.dataset_dir}")
        self.model_path_label.setText(self.openvino_model_path)
        self.output_model_label.setText(self.model_path)
        self.model_file_label.setText(self.model_path)
        self.openvino_file_label.setText(self.openvino_model_path)
        self.attendance_log_label.setText(self.attendance_log_path)

        # Load settings if available (will overwrite defaults if exist)
        self.load_settings()

        # Start camera on initial tab
        self.handle_tab_change(self.tab_widget.currentIndex())

    def handle_tab_change(self, index):
        """Handle tab changes to manage camera resources."""
        # Stop any active camera threads when shifting tabs to avoid conflict
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.stop_recognition()
        
        if self.capture_thread and self.capture_thread.isRunning():
            self.stop_registration()
            
        # If switching to Recognition, do NOT start it automatically
        if index == 0:  # Recognition tab
            self.recognition_status.setText("Ready to start recognition")
                
        # If switching to Registration, start the feed (Preview Mode)
        elif index == 1:  # Registration tab
            self.start_registration_preview()

    def setup_recognition_tab(self):
        recognition_tab = QWidget()
        layout = QVBoxLayout(recognition_tab)

        # Initialize recognition_view first
        self.recognition_view = QLabel()
        self.recognition_view.setAlignment(Qt.AlignCenter)
        self.recognition_view.setStyleSheet("border: 1px solid #cccccc; background-color: #000000;")
        self.recognition_view.setMinimumSize(640, 480)  # Set minimum dimensions
        self.recognition_view.setScaledContents(True)  # Enable image scaling

        # Add to layout
        layout.addWidget(self.recognition_view)

        # Controls
        controls_layout = QHBoxLayout()
        self.start_recognition_btn = QPushButton("Start Recognition")
        self.start_recognition_btn.clicked.connect(self.toggle_recognition)
        controls_layout.addWidget(self.start_recognition_btn)

        self.source_btn = QPushButton("Camera Source")
        self.source_btn.clicked.connect(self.change_camera_source)
        controls_layout.addWidget(self.source_btn)

        layout.addLayout(controls_layout)

        self.recognition_status = QLabel("Recognition inactive")
        self.recognition_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.recognition_status)

        self.tab_widget.addTab(recognition_tab, "Recognition")

    def setup_registration_tab(self):
        registration_tab = QWidget()
        layout = QVBoxLayout(registration_tab)

        # Camera view
        self.registration_view = QLabel()
        self.registration_view.setAlignment(Qt.AlignCenter)
        self.registration_view.setStyleSheet("border: 1px solid #cccccc; background-color: #000000;")
        self.registration_view.setMinimumHeight(300)
        layout.addWidget(self.registration_view)

        # Person details
        form_layout = QFormLayout()

        self.person_name_input = QLineEdit()
        form_layout.addRow("Person Name:", self.person_name_input)

        self.num_images_spinner = QSpinBox()
        self.num_images_spinner.setRange(10, 200)
        self.num_images_spinner.setValue(50)
        form_layout.addRow("Number of Images:", self.num_images_spinner)

        self.registration_dir_label = QLabel(f"Dataset Dir: {self.dataset_dir}")
        self.registration_dir_label.setStyleSheet("color: #333333; font-size: 14px; font-weight: bold; margin-top: 5px;")
        form_layout.addRow(self.registration_dir_label)

        layout.addLayout(form_layout)

        # Progress
        self.registration_progress = QProgressBar()
        layout.addWidget(self.registration_progress)

        # Controls
        controls_layout = QHBoxLayout()

        self.dataset_dir_btn = QPushButton("Select Dataset Directory")
        self.dataset_dir_btn.clicked.connect(self.select_dataset_dir)
        controls_layout.addWidget(self.dataset_dir_btn)

        self.register_btn = QPushButton("Register Face")
        self.register_btn.clicked.connect(self.toggle_registration)
        controls_layout.addWidget(self.register_btn)

        layout.addLayout(controls_layout)

        # Status
        self.registration_status = QLabel("Registration inactive")
        self.registration_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.registration_status)

        self.tab_widget.addTab(registration_tab, "Registration")

    def setup_training_tab(self):
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)

        # Form layout
        form_layout = QFormLayout()

        self.dataset_dir_label = QLabel("Not selected")
        form_layout.addRow("Dataset Directory:", self.dataset_dir_label)

        self.model_path_label = QLabel("Not selected")
        form_layout.addRow("OpenVINO Model:", self.model_path_label)

        self.output_model_label = QLabel("Not selected")
        form_layout.addRow("Output Model:", self.output_model_label)

        layout.addLayout(form_layout)

        # Controls
        controls_layout = QHBoxLayout()

        self.select_dataset_btn = QPushButton("Select Dataset")
        self.select_dataset_btn.clicked.connect(self.select_dataset_dir)
        controls_layout.addWidget(self.select_dataset_btn)

        self.select_model_btn = QPushButton("Select OpenVINO Model")
        self.select_model_btn.clicked.connect(self.select_openvino_model)
        controls_layout.addWidget(self.select_model_btn)

        self.select_output_btn = QPushButton("Select Output Location")
        self.select_output_btn.clicked.connect(self.select_output_model)
        controls_layout.addWidget(self.select_output_btn)

        layout.addLayout(controls_layout)

        # Progress
        self.training_progress = QProgressBar()
        layout.addWidget(self.training_progress)

        # Train button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)

        # Status
        self.training_status = QLabel("Ready to train")
        self.training_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.training_status)

        self.tab_widget.addTab(training_tab, "Training")

    def setup_settings_tab(self):
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)

        # Form layout
        form_layout = QFormLayout()

        # Model settings
        self.model_file_label = QLabel("Not selected")
        form_layout.addRow("Facial Recognition Model:", self.model_file_label)

        self.openvino_file_label = QLabel("Not selected")
        form_layout.addRow("OpenVINO Model:", self.openvino_file_label)

        # Attendance log
        self.attendance_log_label = QLabel("Not selected")
        form_layout.addRow("Attendance Log:", self.attendance_log_label)

        layout.addLayout(form_layout)

        # Controls
        controls_layout = QHBoxLayout()

        self.select_model_file_btn = QPushButton("Select Model File")
        self.select_model_file_btn.clicked.connect(self.select_model_file)
        controls_layout.addWidget(self.select_model_file_btn)

        self.select_openvino_file_btn = QPushButton("Select OpenVINO Model")
        self.select_openvino_file_btn.clicked.connect(self.select_openvino_model)
        controls_layout.addWidget(self.select_openvino_file_btn)

        self.select_log_btn = QPushButton("Select Attendance Log")
        self.select_log_btn.clicked.connect(self.select_attendance_log)
        controls_layout.addWidget(self.select_log_btn)

        layout.addLayout(controls_layout)

        # Save button
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)
        layout.addWidget(self.save_settings_btn)

        self.tab_widget.addTab(settings_tab, "Settings")

    def setup_attendance_tab(self):
        attendance_tab = QWidget()
        layout = QVBoxLayout(attendance_tab)

        # Date filter layout
        date_layout = QHBoxLayout()

        date_label = QLabel("Filter by date:")
        date_layout.addWidget(date_label)

        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        date_layout.addWidget(self.date_edit)

        self.filter_btn = QPushButton("Apply Filter")
        self.filter_btn.clicked.connect(self.filter_attendance)
        date_layout.addWidget(self.filter_btn)

        self.show_all_btn = QPushButton("Show All")
        self.show_all_btn.clicked.connect(self.load_attendance_data)
        date_layout.addWidget(self.show_all_btn)

        layout.addLayout(date_layout)

        # Table widget
        self.attendance_table = QTableWidget(0, 2)
        self.attendance_table.setHorizontalHeaderLabels(["Name", "Timestamp"])
        self.attendance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.attendance_table)

        # Controls
        controls_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_attendance_data)
        controls_layout.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_attendance)
        controls_layout.addWidget(self.export_btn)

        layout.addLayout(controls_layout)

        self.tab_widget.addTab(attendance_tab, "Attendance Log")

    def filter_attendance(self):
        """Filter attendance data by date"""
        abs_log_path = get_path(self.attendance_log_path)
        if not self.attendance_log_path or not os.path.exists(abs_log_path):
            return

        filter_date = self.date_edit.date().toString("yyyy-MM-dd")

        # Clear table
        self.attendance_table.setRowCount(0)

        # Read CSV and populate table with filtered data
        with open(abs_log_path, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header

            row_idx = 0
            for row in reader:
                if len(row) >= 2 and row[1].startswith(filter_date):
                    self.attendance_table.insertRow(row_idx)
                    self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(row[0]))
                    self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(row[1]))
                    row_idx += 1

    def change_camera_source(self):
        dialog = CameraSourceDialog(self)
        if dialog.exec_():
            self.camera_source = dialog.get_camera_source()
            QMessageBox.information(self, "Camera Source", f"Camera source set to: {self.camera_source}")

            # Update camera source in active threads
            if self.recognition_thread and self.recognition_thread.isRunning():
                self.recognition_thread.stop()
                self.start_recognition()

            if self.capture_thread and self.capture_thread.isRunning():
                self.capture_thread.stop()
                self.start_registration()

    def toggle_recognition(self):
        if not self.recognition_thread or not self.recognition_thread.isRunning():
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model file in Settings")
            return

        if not os.path.exists(self.openvino_model_path):
            QMessageBox.warning(self, "Error", "Please select a valid OpenVINO model in Settings")
            return

        # Create and set up recognition thread
        self.recognition_thread = FaceRecognitionThread(self)
        self.recognition_thread.set_camera_source(self.camera_source)
        self.recognition_thread.load_model(self.model_path, self.openvino_model_path)
        self.recognition_thread.update_frame.connect(self.update_recognition_frame)

        # Update UI
        self.recognition_thread.start()
        self.start_recognition_btn.setText("Stop Recognition")
        self.recognition_status.setText("Recognition active")

    def stop_recognition(self):
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.stop()
            self.start_recognition_btn.setText("Start Recognition")
            
            # Reset view to black with a neat centered message
            self.recognition_view.clear()
            self.recognition_view.setText("Camera Feed Stopped\nClick 'Start Recognition' to begin")
            self.recognition_view.setAlignment(Qt.AlignCenter)
            self.recognition_view.setStyleSheet("""
                border: 2px solid #333333; 
                background-color: #000000; 
                color: #ffffff; 
                font-size: 16px; 
                font-weight: bold;
            """)
            self.recognition_status.setText("Recognition inactive")

    def update_recognition_frame(self, frame, recognized_faces):
        # Check for empty frames
        if frame is None or frame.size == 0:
            return

        # Draw rectangles and names for recognized faces
        for name, (x, y, w, h), confidence in recognized_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            confidence_text = f"{confidence:.2f}"
            cv2.putText(frame, f"{name} ({confidence_text})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Log attendance
            if self.attendance_log_path:
                self.log_attendance(name)

        # Convert frame to QImage and display
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width

            # Fix color format conversion (BGR -> RGB)
            q_img = QImage(frame.data, width, height, bytes_per_line,
                           QImage.Format_BGR888).rgbSwapped()

            # Get current label dimensions
            label_width = self.recognition_view.width()
            label_height = self.recognition_view.height()

            # Handle initial zero-size labels
            if label_width <= 0 or label_height <= 0:
                label_width = 640
                label_height = 480

            # Maintain aspect ratio
            pixmap = QPixmap.fromImage(q_img).scaled(
                label_width, label_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Center the pixmap in the label
            self.recognition_view.setPixmap(pixmap)
            self.recognition_view.setAlignment(Qt.AlignCenter)

        except Exception as e:
            print(f"Frame display error: {str(e)}")

    def toggle_registration(self):
        """Modified toggle to only start/stop capture, assuming feed is active."""
        if self.capture_thread and self.capture_thread.isRunning():
            if not self.capture_thread.capture_active:
                person_name = self.person_name_input.text().strip()
                if not person_name:
                    QMessageBox.warning(self, "Error", "Please enter a person name")
                    return
                if not self.dataset_dir:
                    QMessageBox.warning(self, "Error", "Please select a dataset directory")
                    return

                self.capture_thread.set_capture_params(
                    self.dataset_dir,
                    person_name,
                    self.num_images_spinner.value()
                )
                self.registration_progress.setValue(0)
                self.capture_thread.start_capture()
                self.register_btn.setText("Stop Capture")
                self.registration_status.setText(f"Capturing {person_name}...")
            else:
                self.capture_thread.stop_capture()
                self.register_btn.setText("Register Face")
                self.registration_status.setText("Capture stopped")

    def start_registration_preview(self):
        """Starts the camera feed for the registration tab without capturing."""
        if not self.capture_thread or not self.capture_thread.isRunning():
            self.capture_thread = FaceCapturingThread(self)
            self.capture_thread.set_camera_source(self.camera_source)
            self.capture_thread.update_frame.connect(self.update_registration_frame)
            self.capture_thread.update_progress.connect(self.registration_progress.setValue)
            self.capture_thread.capture_complete.connect(self.on_registration_complete)
            self.capture_thread.start()
            self.registration_status.setText("Camera feed active")

    def start_registration(self):
        # Legacy method, now handled by preview + toggle
        self.start_registration_preview()

    def stop_registration(self):
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.stop()
            self.register_btn.setText("Register Face")
            
            # Reset view to black with a neat centered message
            self.registration_view.clear()
            self.registration_view.setText("Camera Feed Stopped\nSwitch tabs or click 'Register Face' to resume")
            self.registration_view.setAlignment(Qt.AlignCenter)
            self.registration_view.setStyleSheet("""
                border: 2px solid #333333; 
                background-color: #000000; 
                color: #ffffff; 
                font-size: 16px; 
                font-weight: bold;
            """)
            self.registration_status.setText("Camera feed inactive")

    def update_registration_frame(self, frame):
        # Convert frame to QImage and display
        if frame is None or frame.size == 0:
            return
        
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            
            # Use a more robust BGR to RGB conversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Maintain aspect ratio and scale
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.registration_view.width(), self.registration_view.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.registration_view.setPixmap(pixmap)
        except Exception as e:
            print(f"Registration preview error: {str(e)}")

    def on_registration_complete(self):
        self.register_btn.setText("Register Face")
        self.registration_status.setText("Registration complete!")
        # Stop capture mode but keep the feed running
        if self.capture_thread:
            self.capture_thread.stop_capture()
        
        QMessageBox.information(self, "Registration Complete",
                                f"Successfully captured faces for {self.person_name_input.text()}")

    def start_training(self):
        if not self.dataset_dir:
            QMessageBox.warning(self, "Error", "Please select a dataset directory")
            return

        if not self.openvino_model_path:
            QMessageBox.warning(self, "Error", "Please select an OpenVINO model")
            return

        if not self.model_path:
            QMessageBox.warning(self, "Error", "Please select an output model location")
            return

        # Create and set up training thread
        self.training_thread = ModelTrainingThread(self)
        self.training_thread.set_training_params(
            self.dataset_dir,
            self.openvino_model_path,
            self.model_path
        )
        self.training_thread.update_progress.connect(self.update_training_progress)
        self.training_thread.training_complete.connect(self.on_training_complete)

        # Update UI
        self.training_progress.setValue(0)
        self.train_btn.setEnabled(False)
        self.training_status.setText("Training in progress...")
        self.training_thread.start()

    def update_training_progress(self, progress, status_text):
        self.training_progress.setValue(progress)
        self.training_status.setText(status_text)

    def on_training_complete(self):
        self.train_btn.setEnabled(True)
        QMessageBox.information(self, "Training Complete",
                                "Model training completed successfully!")

    def select_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", BASE_DIR)
        if dir_path:
            # Store relative path if it's within BASE_DIR
            if dir_path.startswith(BASE_DIR):
                rel_path = os.path.relpath(dir_path, BASE_DIR)
            else:
                rel_path = dir_path
            
            self.dataset_dir = rel_path
            self.dataset_dir_label.setText(rel_path)
            if hasattr(self, 'registration_dir_label'):
                self.registration_dir_label.setText(f"Dataset Dir: {rel_path}")

    def select_openvino_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select OpenVINO Model", BASE_DIR, "XML Files (*.xml)")
        if file_path:
            if file_path.startswith(BASE_DIR):
                rel_path = os.path.relpath(file_path, BASE_DIR)
            else:
                rel_path = file_path
                
            self.openvino_model_path = rel_path
            self.openvino_file_label.setText(rel_path)
            self.model_path_label.setText(rel_path)

    def select_output_model(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model As", BASE_DIR, "Pickle Files (*.pkl)")
        if file_path:
            if file_path.startswith(BASE_DIR):
                rel_path = os.path.relpath(file_path, BASE_DIR)
            else:
                rel_path = file_path
            self.model_path = rel_path
            self.output_model_label.setText(rel_path)

    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", BASE_DIR, "Pickle Files (*.pkl)")
        if file_path:
            if file_path.startswith(BASE_DIR):
                rel_path = os.path.relpath(file_path, BASE_DIR)
            else:
                rel_path = file_path
            self.model_path = rel_path
            self.model_file_label.setText(rel_path)

    def select_attendance_log(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Attendance Log", BASE_DIR, "CSV Files (*.csv)")
        if file_path:
            if file_path.startswith(BASE_DIR):
                rel_path = os.path.relpath(file_path, BASE_DIR)
            else:
                rel_path = file_path
            self.attendance_log_path = rel_path
            self.attendance_log_label.setText(rel_path)

            # Create the file if it doesn't exist (using absolute path for file ops)
            abs_path = get_path(rel_path)
            if not os.path.exists(abs_path):
                with open(abs_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Name", "Timestamp"])

            # Load attendance data
            self.load_attendance_data()

    def log_attendance(self, name):
        if not self.attendance_log_path:
            return

        abs_log_path = get_path(self.attendance_log_path)

        # Check if file exists, create with headers if not
        if not os.path.exists(abs_log_path):
            with open(abs_log_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Timestamp"])

        # Check if this person was already logged today
        today = datetime.now().strftime("%Y-%m-%d")
        already_logged = False

        with open(abs_log_path, 'r', newline='') as file:
            reader = csv.reader(file)
            try:
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2 and row[0] == name and row[1].startswith(today):
                        already_logged = True
                        break
            except StopIteration:
                pass

        # Log attendance if not already logged today
        if not already_logged:
            with open(abs_log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            self.load_attendance_data()  # Refresh the table

    def load_attendance_data(self):
        """Load attendance data from CSV file to table widget"""
        abs_log_path = get_path(self.attendance_log_path)
        if not self.attendance_log_path or not os.path.exists(abs_log_path):
            return

        # Clear table
        self.attendance_table.setRowCount(0)

        # Read CSV and populate table
        with open(abs_log_path, 'r', newline='') as file:
            reader = csv.reader(file)
            try:
                next(reader)  # Skip header
                for row_idx, row in enumerate(reader):
                    if len(row) >= 2:
                        self.attendance_table.insertRow(row_idx)
                        self.attendance_table.setItem(row_idx, 0, QTableWidgetItem(row[0]))
                        self.attendance_table.setItem(row_idx, 1, QTableWidgetItem(row[1]))
            except StopIteration:
                pass

        # Sort by timestamp descending
        self.attendance_table.sortItems(1, Qt.DescendingOrder)

    def export_attendance(self):
        """Export attendance data to a new CSV file"""
        if not self.attendance_log_path or not os.path.exists(self.attendance_log_path):
            QMessageBox.warning(self, "Error", "No attendance data available")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Attendance", "", "CSV Files (*.csv)")
        if file_path:
            try:
                abs_log_path = get_path(self.attendance_log_path)
                with open(abs_log_path, 'r', newline='') as src_file:
                    with open(file_path, 'w', newline='') as dst_file:
                        reader = csv.reader(src_file)
                        writer = csv.writer(dst_file)
                        for row in reader:
                            writer.writerow(row)

                QMessageBox.information(self, "Export Complete", "Attendance data exported successfully")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")

    def save_settings(self):
        """Save application settings (storing relative paths)"""
        settings = {
            "dataset_dir": self.dataset_dir,
            "model_path": self.model_path,
            "openvino_model_path": self.openvino_model_path,
            "attendance_log_path": self.attendance_log_path
        }

        try:
            with open(get_path("facial_attendance_settings.pkl"), "wb") as f:
                pickle.dump(settings, f)
            QMessageBox.information(self, "Settings Saved", "Application settings saved successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings: {str(e)}")

    def load_settings(self):
        """Load application settings and update UI"""
        try:
            settings_path = get_path("facial_attendance_settings.pkl")
            if os.path.exists(settings_path):
                with open(settings_path, "rb") as f:
                    settings = pickle.load(f)

                self.dataset_dir = settings.get("dataset_dir", "")
                self.model_path = settings.get("model_path", "")
                self.openvino_model_path = settings.get("openvino_model_path", "")
                self.attendance_log_path = settings.get("attendance_log_path", "")

                # Update UI with relative paths
                self.dataset_dir_label.setText(self.dataset_dir)
                if hasattr(self, 'registration_dir_label'):
                    self.registration_dir_label.setText(f"Dataset Dir: {self.dataset_dir}")
                self.model_path_label.setText(self.openvino_model_path)
                self.output_model_label.setText(self.model_path)
                self.model_file_label.setText(self.model_path)
                self.openvino_file_label.setText(self.openvino_model_path)
                self.attendance_log_label.setText(self.attendance_log_path)

                # Load attendance data
                self.load_attendance_data()
        except Exception as e:
            print(f"Error loading settings: {str(e)}")

    def closeEvent(self, event):
        """Clean up threads before closing"""
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.stop()

        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.stop()

        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.wait()

        event.accept()


# Main application entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look and feel

    # Set application icon and styling
    app.setWindowIcon(QIcon("icon.png"))  # You'll need to create an icon file

    # Apply some basic styling
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QPushButton {
            background-color: #0078d7;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 2px;
        }
        QPushButton:hover {
            background-color: #005a9e;
        }
        QPushButton:pressed {
            background-color: #004275;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #e1e1e1;
            padding: 8px 12px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-top: 2px solid #0078d7;
        }
        QTableWidget {
            gridline-color: #d0d0d0;
            selection-background-color: #0078d7;
        }
    """)

    window = FacialAttendanceSystem()
    window.show()
    sys.exit(app.exec_())