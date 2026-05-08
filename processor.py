import cv2
from mtcnn import MTCNN
import numpy as np

# Initialize MTCNN detector
# ما از این مدل برای تشخیص دقیق اجزای صورت استفاده می‌کنیم
detector = MTCNN()

def extract_face_sequence(video_path, sequence_length=10):
    """
    Extracts a sequence of normalized face images from a video.
    """
    cap = cv2.VideoCapture(video_path)
    face_sequence = []
    
    # برای حفظ سرعت، فریم‌ها را با فاصله برمی‌داریم
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // sequence_length)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret or len(face_sequence) >= sequence_length:
            break
            
        # Convert to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)
        
        if results:
            # گرفتن بزرگترین چهره در تصویر
            res = max(results, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = res['box']
            
            # Crop and Resize to 224x224 (Standard for CNN/ViT)
            face = rgb_frame[max(0, y):y+h, max(0, x):x+w]
            face = cv2.resize(face, (224, 224))
            
            # Normalization (0 to 1)
            face_sequence.append(face.astype('float32') / 255.0)

    cap.release()
    return np.array(face_sequence)