import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN

class FastMTCNN:
    def __init__(self, stride=4, resize=0.5, margin=14, factor=0.6, keep_all=True, device=None):
        """
        FastMTCNN: Tăng tốc nhận diện khuôn mặt bằng cách sử dụng batch processing và tối ưu hóa GPU.

        - stride: Khoảng cách trượt của detector (giảm để tăng tốc)
        - resize: Tỷ lệ thu nhỏ ảnh trước khi phát hiện khuôn mặt (giảm để xử lý nhanh hơn)
        - margin: Kích thước lề xung quanh khuôn mặt
        - factor: Scale factor khi phát hiện nhiều kích thước khuôn mặt
        - keep_all: Giữ lại tất cả khuôn mặt trong ảnh hay chỉ lấy khuôn mặt chính
        - device:dùng cuda nếu có GPU, nếu không sẽ tự động sử dụng CPU
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.resize = resize
        self.mtcnn = MTCNN(
            margin=margin,
            factor=factor,
            keep_all=keep_all,
            device=self.device
        )

    def detect(self, img):
        """
        Phát hiện khuôn mặt trong ảnh và trả về tọa độ bounding box + confidence scores.
        """
        # Resize ảnh để tăng tốc độ xử lý
        height, width, _ = img.shape
        small_img = cv2.resize(img, (int(width * self.resize), int(height * self.resize)))

        # Chuyển ảnh sang RGB (MTCNN yêu cầu)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        # Phát hiện khuôn mặt
        boxes, probs = self.mtcnn.detect(rgb_small_img)

        if boxes is not None:
            # Scale bounding box trở lại kích thước ban đầu
            boxes /= self.resize

        return boxes, probs

    def extract_faces(self, img):
        """
        Cắt và trích xuất các khuôn mặt từ ảnh.
        """
        boxes, probs = self.detect(img)

        faces = []
        if boxes is not None:
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box)
                face = img[y_min:y_max, x_min:x_max]
                faces.append(face)

        return faces, probs
