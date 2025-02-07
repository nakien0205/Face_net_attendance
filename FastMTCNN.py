import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN

class FastMTCNN:
    def __init__(self, stride=4, resize=0.5, margin=14, factor=0.6, keep_all=True, device=None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.resize = resize
        self.mtcnn = MTCNN(
            margin=margin,
            factor=factor,
            keep_all=keep_all,
            device=self.device
        )

    def detect(self, img):

        height, width, _ = img.shape
        small_img = cv2.resize(img, (int(width * self.resize), int(height * self.resize)))
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(rgb_small_img)

        if boxes is not None:
            boxes /= self.resize

        return boxes, probs

    def extract_faces(self, img):

        boxes, probs = self.detect(img)

        faces = []
        if boxes is not None:
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box)
                face = img[y_min:y_max, x_min:x_max]
                faces.append(face)

        return faces, probs
