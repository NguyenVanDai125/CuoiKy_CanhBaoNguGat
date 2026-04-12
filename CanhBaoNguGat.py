import cv2
import mediapipe as mp
import winsound

# Cấu hình MediaPipe
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Thiết lập AI (Cần có file 'face_landmarker.task' cùng thư mục)
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)

# Biến đếm để lọc chớp mắt
COUNTER = 0
THRESHOLD_FRAMES = 15  # Ngưỡng: nhắm mắt liên tục ~0.5s sẽ cảnh báo

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    print("Hệ thống đã sẵn sàng! Nhấn 'q' để thoát.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Lật ảnh để hiệu ứng gương
        frame = cv2.flip(frame, 1)
        
        # Chuyển đổi định dạng cho MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        
        # Logic nhận diện
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            # Lấy tọa độ mí mắt trên (159) và dưới (145)
            top_eye = landmarks[159].y
            bottom_eye = landmarks[145].y
            
            # Kiểm tra khoảng cách
            if abs(top_eye - bottom_eye) < 0.012: # Tăng/giảm 0.012 nếu cần nhạy hơn
                COUNTER += 1
                if COUNTER >= THRESHOLD_FRAMES:
                    cv2.putText(frame, "CANH BAO NGU GAT!", (50, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    winsound.Beep(1000, 200)
            else:
                COUNTER = 0 # Reset nếu mở mắt

        cv2.imshow('Drowsiness Detection - Dai IT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()