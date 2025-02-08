from facenet_pytorch import MTCNN
import torch
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FastMTCNN:
    def __init__(self, margin, factor, keep_all=False, device=device, thresholds=[0.6, 0.7, 0.7]):

        self.mtcnn = MTCNN(margin=margin, factor=factor, keep_all=keep_all, device=device, thresholds=thresholds)

    def detect(self, img):
        # Downscale image if its largest dimension exceeds max_dim
        height, width = img.shape[:2]
        max_dim = 800  # maximum dimension (adjust as needed)
        scale = 1.0
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img_resized = cv2.resize(img, (int(width * scale), int(height * scale)))
        else:
            img_resized = img

        # Use the underlying MTCNN detector on the (possibly resized) image.
        bboxes, probs = self.mtcnn.detect(img_resized)

        # If the image was resized, scale bounding boxes back to the original size.
        if bboxes is not None and scale != 1.0:
            bboxes = bboxes / scale

        return bboxes, probs
