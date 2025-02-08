import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse


class FaceRecognizer:
    def __init__(self, dataset_path='face_test', device=None):
        # Use CUDA if available
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize MTCNN for multi-face detection (keep_all=True returns all faces)
        self.mtcnn = MTCNN(keep_all=True, device=self.device, thresholds=[0.7, 0.7, 0.7])
        # Load InceptionResnetV1 pretrained on VGGFace2 for computing face embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # Dictionary to store each student's average embedding
        self.database = {}
        self.dataset_path = dataset_path

    def build_database(self, detection_threshold=0.95):
        """
        Traverse the dataset directory structured as:
            face_test / classroom / student / image files
        For each student, compute the embedding from each image (if a face is detected
        with a probability above detection_threshold) and average them.
        """
        print("Building face database from dataset...")
        for classroom in os.listdir(self.dataset_path):
            classroom_path = os.path.join(self.dataset_path, classroom)
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
                    # For enrollment, we assume one face per image
                    face, prob = self.mtcnn(img, return_prob=True)
                    if face is None:
                        print(f"No face detected in {img_path}")
                        continue
                    if prob < detection_threshold:
                        print(f"Low detection probability ({prob:.2f}) in {img_path}")
                        continue
                    # The detected face is a tensor of shape (3, 160, 160)
                    face = face.unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        emb = self.resnet(face)
                    embeddings.append(emb.cpu().numpy())
                if embeddings:
                    # Average the embeddings for a robust representation
                    avg_embedding = np.mean(embeddings, axis=0)
                    student_id = f"{classroom}_{student}"
                    self.database[student_id] = avg_embedding
                    print(f"Added {student_id} with {len(embeddings)} images.")
                else:
                    print(f"No valid images for {classroom}/{student}.")
        print("Database building complete.")

    def recognize_image(self, img_path, threshold=0.8):
        """
        Process a given image to:
          1. Detect all faces and obtain bounding boxes and detection probabilities.
          2. For each detected face, compute its embedding.
          3. Compare each embedding to the database (using cosine similarity) to identify the student.
          4. Among all detections, return the bounding box (and associated info) with the highest detection confidence.

        Args:
          img_path (str): Path to the input image.
          threshold (float): Cosine similarity threshold; if the best match is below this, label as "unknown".

        Returns:
          dict: A dictionary containing "bounding_box", "identity", and "confidence".
        """
        img = Image.open(img_path).convert('RGB')
        # First detect faces and get bounding boxes and detection probabilities
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            print("No faces detected in the image.")
            return None

        # Crop all detected faces (MTCNN internally resizes them to 160x160)
        faces, _ = self.mtcnn(img, return_prob=True)
        if faces is None or len(faces) == 0:
            print("Failed to crop faces from the image.")
            return None

        best_confidence = -1
        best_bbox = None
        best_identity = None

        # Process each detected face
        for i, face in enumerate(faces):
            face_tensor = face.unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.resnet(face_tensor)
            emb_vec = emb.cpu().numpy().flatten()

            # Compare against all enrolled embeddings
            best_sim = -1
            candidate = None
            for identity, db_emb in self.database.items():
                db_emb_vec = db_emb.flatten()
                # Compute cosine similarity between the two embeddings
                cos_sim = np.dot(emb_vec, db_emb_vec) / (np.linalg.norm(emb_vec) * np.linalg.norm(db_emb_vec))
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    candidate = identity

            # Assign label if the similarity exceeds the threshold; otherwise, mark as "unknown"
            label = candidate if best_sim >= threshold else "unknown"
            # Get detection confidence from MTCNN (corresponding to the bounding box)
            prob = probs[i] if probs is not None else 0
            # Retain the face detection with the highest confidence score
            if prob > best_confidence:
                best_confidence = prob
                best_bbox = boxes[i]
                best_identity = label

        return {"bounding_box": best_bbox, "identity": best_identity, "confidence": best_confidence}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Facial Recognition using MTCNN and InceptionResnetV1 with CUDA support"
    )
    parser.add_argument("--dataset", type=str, default="face_test", help="Path to dataset folder")
    parser.add_argument("--image", type=str, help="Path to image file for recognition")
    parser.add_argument("--threshold", type=float, default=0.8, help="Cosine similarity threshold for recognition")
    args = parser.parse_args()

    # Initialize the recognizer (models automatically move to CUDA if available)
    recognizer = FaceRecognizer(dataset_path=args.dataset)
    recognizer.build_database()

    if args.image:
        result = recognizer.recognize_image(args.image, threshold=args.threshold)
        if result:
            print("Best Detection:")
            print(f"Bounding Box: {result['bounding_box']}")
            print(f"Identity: {result['identity']}")
            print(f"Detection Confidence: {result['confidence']}")
    else:
        print("No image provided for recognition. Exiting.")
