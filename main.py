import os
import cv2
import torch
import face_recognition
from facenet_pytorch import MTCNN
import numpy as np
import pickle
import logging
from FastMTCNN import FastMTCNN
import csv
import atexit
import time
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global configurations
BASE_PATH = 'face_test'
CACHE_FILE = 'face_encodings_cache.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.9
TOLERANCE = 0.5
RESIZE_SCALE = 0.5  # Reduce img to 50%
FRAME_SKIP = 3  #Process every 3 frames to reduce CPU load
MTCNN_THRESHOLDS = [0.7, 0.8, 0.8] # reduce False Positive but will miss some face
attendance_record = {}

# Initialize MTCNN for face detection   
mtcnn = FastMTCNN(device=DEVICE, margin=10, keep_all=False)

def load_known_faces(base_path):
    """
    Loads and encodes faces from the dataset. Each subfolder in base_path represents one user.
    Returns:
        known_encodings (list): List of face encodings.
        known_names (list): Corresponding list of usernames.
    """
    known_encodings = []
    known_names = []

    for user_name in os.listdir(base_path):
        user_folder = os.path.join(base_path, user_name)
        if os.path.isdir(user_folder):
            logging.info(f"Processing images for user: {user_name}")
            for img_file in os.listdir(user_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(user_folder, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        logging.warning(f"Failed to load image: {img_path}")
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bboxes, probs = mtcnn.detect(img_rgb)
                    if bboxes is not None:
                        # Select the face with the highest confidence if multiple faces are detected
                        best_idx = np.argmax(probs)
                        best_bbox = bboxes[best_idx]
                        x_min, y_min, x_max, y_max = [int(coord) for coord in best_bbox]
                        face_location = (y_min, x_max, y_max, x_min)  # Format for face_recognition

                        encoding = face_recognition.face_encodings(img_rgb, [face_location])
                        if encoding:
                            known_encodings.append(encoding[0])
                            known_names.append(user_name)
                        else:
                            logging.warning(f"Face encoding not found in image: {img_path}")
                    else:
                        logging.warning(f"No face detected in {img_path}")
    logging.info(f"Total encodings loaded: {len(known_encodings)}")
    return known_encodings, known_names


def load_known_faces_cached(base_path, cache_file):
    """
    Loads face encodings and names from a cache file if it exists.
    If the cache file does not exist or loading fails, it will process the dataset and then save the encodings.
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Loaded {len(data['encodings'])} encodings from cache.")
            return data['encodings'], data['names']
        except Exception as e:
            logging.error(f"Error loading cache file: {e}. Recomputing encodings.")

    # If cache file doesn't exist or fails to load, compute encodings
    known_encodings, known_names = load_known_faces(base_path)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
        logging.info("Encodings saved to cache.")
    except Exception as e:
        logging.error(f"Error saving cache file: {e}")

    return known_encodings, known_names



def process_frame(frame, known_encodings, known_names, resize_scale=0.25):
    """
    Detects and recognizes faces in the given frame.
    """

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    bboxes, probs = mtcnn.detect(rgb_small_frame)
    if bboxes is not None:
        for bbox, prob in zip(bboxes, probs):
            if prob is None or prob < CONFIDENCE_THRESHOLD:
                continue

            x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
            face_location = (y_min, x_max, y_max, x_min)
            encoding = face_recognition.face_encodings(rgb_small_frame, [face_location])
            if encoding:
                encoding = encoding[0]
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_match_index = np.argmin(distances) if distances.size > 0 else None

                if best_match_index is not None and matches[best_match_index]:
                    detected_name = known_names[best_match_index]
                    color = (0, 255, 0)
                    if detected_name not in attendance_record:
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        attendance_record[detected_name] = {'status': 'Present', 'time': current_time}

                else:
                    detected_name = "Unknown"
                    color = (0, 0, 255)

                # Draw bounding box (scaling coordinates back to original frame)
                scale_factor = int(1 / resize_scale)
                cv2.rectangle(frame,
                              (x_min * scale_factor, y_min * scale_factor),
                              (x_max * scale_factor, y_max * scale_factor),
                              color, 2)
                cv2.putText(frame, f"{detected_name} ({prob:.2f})",
                            (x_min * scale_factor, y_min * scale_factor - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame



def save_attendance_csv():
    """Lưu danh sách điểm danh vào file CSV"""
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Status", "Time"])
        for name, info in attendance_record.items():
            writer.writerow([name, info["status"], info["time"]])
    print("Danh sách điểm danh đã được lưu vào attendance.csv")


def main():
    # Load known face encodings from cache or process the dataset if cache is not available.
    known_encodings, known_names = load_known_faces_cached(BASE_PATH, CACHE_FILE)
    if not known_encodings:
        logging.error("No known face encodings found. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Unable to access webcam. Exiting.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame. Exiting.")
                break

            frame = process_frame(frame, known_encodings, known_names)
            cv2.imshow("Real-Time Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception:
        logging.exception("An error occurred during processing:")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":
    main()
    atexit.register(save_attendance_csv)


