import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognizer:
    def __init__(self, base_dataset='face_test', database_file='face_database.pt', device=None):
        """
        Args:
            base_dataset (str): Path to the dataset folder that contains 'train' and 'test' subfolders.
            database_file (str): File name for saving/loading the computed face embeddings.
            device: Torch device (defaults to CUDA if available).
        """
        self.base_dataset = base_dataset
        self.train_path = os.path.join(base_dataset, 'train')
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Initialize MTCNN for multi-face detection. keep_all=True returns all detected faces.
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        # Load the InceptionResnetV1 (pretrained on VGGFace2) to compute embeddings.
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # Dictionary to store each student's average embedding.
        self.database = {}
        self.database_file = database_file

    def build_database(self, detection_threshold=0.95):
        """
        Build a face database from the training data.
        Expected folder structure:
            base_dataset/train / classroom / student / image files
        For each student, the function detects a face from each image (if the detection probability
        is above detection_threshold) and averages the embeddings to form a robust representation.
        """
        print("Building face database from training data...")
        for classroom in os.listdir(self.train_path):
            classroom_path = os.path.join(self.train_path, classroom)
            if not os.path.isdir(classroom_path):
                continue
            for student in os.listdir(classroom_path):
                student_path = os.path.join(classroom_path, student)
                if not os.path.isdir(student_path):
                    continue
                embeddings = []
                for img_name in os.listdir(student_path):
                    img_path = os.path.join(student_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        print(f"Could not open image {img_path}: {e}")
                        continue
                    # Detect face and return its probability.
                    face, prob = self.mtcnn(img, return_prob=True)
                    if face is None:
                        print(f"No face detected in {img_path}. Skipping.")
                        continue
                    for face, prob in zip(face, prob):
                        if prob < detection_threshold:
                            print(f"Low detection probability ({prob:.2f}) in {img_path}. Skipping.")
                            continue
                    # Add batch dimension: face tensor shape becomes (1, 3, 160, 160)
                    face = face.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        emb = self.resnet(face)
                    embeddings.append(emb.cpu().numpy())
                if embeddings:
                    # Average the embeddings for a robust representation.
                    avg_embedding = np.mean(embeddings, axis=0)
                    student_id = f"{classroom}_{student}"
                    self.database[student_id] = avg_embedding
                    print(f"Added {student_id} with {len(embeddings)} images.")
                else:
                    print(f"No valid images for {classroom}/{student}.")
        print("Database building complete.")

    def save_database(self):
        """Save the computed face database to disk using torch.save."""
        torch.save(self.database, self.database_file)
        print(f"Database saved to {self.database_file}")

    def load_database(self):
        """Load the face database from disk if it exists.

        Returns:
            bool: True if the database was loaded, False otherwise.
        """
        if os.path.exists(self.database_file):
            self.database = torch.load(self.database_file)
            print(f"Loaded database from {self.database_file}")
            return True
        return False

    def recognize_image(self, img_path, threshold=0.8):
        """
        Process a given image to:
          1. Detect all faces and obtain bounding boxes and detection probabilities.
          2. For each detected face, compute its embedding.
          3. Compare each embedding to the stored database (using cosine similarity) to identify the student.
          4. Among all detections, return the bounding box (and associated info) with the highest detection confidence.

        Args:
          img_path (str): Path to the input image.
          threshold (float): Cosine similarity threshold; if the best match is below this, label as "unknown".

        Returns:
          dict: A dictionary containing "bounding_box", "identity", and "confidence", or None if no face is detected.
        """
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            return None

        # First detect faces and get bounding boxes and probabilities.
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            print(f"No faces detected in {img_path}.")
            return None

        # Crop all detected faces (the detector also returns resized tensors).
        faces, _ = self.mtcnn(img, return_prob=True)
        if faces is None or len(faces) == 0:
            print(f"Failed to crop faces from {img_path}.")
            return None

        best_confidence = -1
        best_bbox = None
        best_identity = None

        # Process each detected face.
        for i, face in enumerate(faces):
            face_tensor = face.unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.resnet(face_tensor)
            emb_vec = emb.cpu().numpy().flatten()

            # Compare against all enrolled embeddings.
            best_sim = -1
            candidate = None
            for identity, db_emb in self.database.items():
                db_emb_vec = db_emb.flatten()
                # Compute cosine similarity.
                cos_sim = np.dot(emb_vec, db_emb_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(db_emb_vec))
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    candidate = identity

            # Assign label if the similarity exceeds the threshold; otherwise, mark as "unknown".
            label = candidate if best_sim >= threshold else "unknown"
            prob = probs[i] if probs is not None else 0
            if prob > best_confidence:
                best_confidence = prob
                best_bbox = boxes[i]
                best_identity = label

        return {"bounding_box": best_bbox, "identity": best_identity, "confidence": best_confidence}

    def validate(self, threshold=0.8):
        """
        Validate the recognition system on test data.
        Expected folder structure for testing:
            base_dataset/test / classroom / student / image files
        For each test image, the predicted identity (from recognize_image) is compared to the
        ground truth (constructed as "classroom_student"). Overall accuracy is reported.

        Args:
            threshold (float): Cosine similarity threshold for recognition.
        """
        test_path = os.path.join(self.base_dataset, 'test')
        total = 0
        correct = 0
        print("Starting validation on test dataset...")
        for classroom in os.listdir(test_path):
            classroom_path = os.path.join(test_path, classroom)
            if not os.path.isdir(classroom_path):
                continue
            for student in os.listdir(classroom_path):
                student_path = os.path.join(classroom_path, student)
                if not os.path.isdir(student_path):
                    continue
                expected_label = f"{classroom}_{student}"
                for img_name in os.listdir(student_path):
                    img_path = os.path.join(student_path, img_name)
                    total += 1
                    result = self.recognize_image(img_path, threshold=threshold)
                    if result is None:
                        print(f"Image {img_path}: No face detected.")
                        continue
                    predicted_label = result["identity"]
                    if predicted_label == expected_label:
                        correct += 1
                    else:
                        print(f"Mismatch for {img_path}: expected {expected_label}, got {predicted_label}")
        accuracy = correct / total if total > 0 else 0
        print(f"Validation complete: {correct}/{total} correct. Accuracy: {accuracy:.2f}")


def main():
    # Create an instance of FaceRecognizer
    recognizer = FaceRecognizer(base_dataset="face_test", database_file="face_database.pt")

    # Load the saved database if it exists; otherwise, build and save a new one.
    if not recognizer.load_database():
        recognizer.build_database()
        recognizer.save_database()

    # Call functions directly as desired:
    #
    # To recognize a single image, set the image path below:
    image_path = r'D:\Python\Projects\CV\Computer Vision\FaceNet\face_test\train\AI1904\Kien\z6256184631289_25e86b3fa079e7b958c58431b6b54c2d.jpg'
    result = recognizer.recognize_image(image_path, threshold=0.8)
    if result:
        print("Best Detection:")
        print("Bounding Box:", result["bounding_box"])
        print("Identity:", result["identity"])
        print("Detection Confidence:", result["confidence"])
    else:
        print("No valid face detected in the provided image.")

    # To run validation on your test dataset, simply call:
    # recognizer.validate(threshold=0.8)


if __name__ == '__main__':
    main()
