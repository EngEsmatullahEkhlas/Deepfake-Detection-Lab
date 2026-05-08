import cv2
from mtcnn import MTCNN
import os

def extract_faces_from_video(video_path, output_folder, sample_rate=10):
    # ایجاد پوشه خروجی اگر وجود ندارد
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # مقداردهی اولیه آشکارساز چهره
    detector = MTCNN()
    
    # باز کردن فایل ویدیو
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    print("در حال پردازش ویدیو و استخراج چهره‌ها...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # پردازش فریم‌ها بر اساس نرخ نمونه‌برداری (مثلاً هر ۱۰ فریم یکبار)
        if frame_count % sample_rate == 0:
            # تبدیل رنگ به RGB (چون MTCNN با RGB کار می‌کند)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # تشخیص چهره
            results = detector.detect_faces(rgb_frame)
            
            for i, result in enumerate(results):
                x, y, w, h = result['box']
                # اطمینان از اینکه مختصات داخل کادر هستند
                x, y = max(0, x), max(0, y)
                
                # برش چهره (Face Cropping)
                face = frame[y:y+h, x:x+w]
                
                if face.size > 0:
                    # تغییر اندازه به مقدار استاندارد (مثلاً 224x224)
                    face_resized = cv2.resize(face, (224, 224))
                    
                    # ذخیره تصویر چهره
                    face_filename = f"face_{saved_count}.jpg"
                    cv2.imwrite(os.path.join(output_folder, face_filename), face_resized)
                    saved_count += 1

        frame_count += 1
        if saved_count % 10 == 0:
            print(f"تا کنون {saved_count} چهره استخراج شده است...")

    cap.release()
    print(f"پایان عملیات. مجموعاً {saved_count} تصویر چهره در پوشه '{output_folder}' ذخیره شد.")

# --- نحوه استفاده ---
# مسیر ویدیوی خود را اینجا جایگزین کنید
# video_file = "test_video.mp4" 
# extract_faces_from_video(video_file, "extracted_faces")