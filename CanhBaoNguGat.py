import cv2
import mediapipe as mp
import winsound
import threading
import os
import math

# --- Cấu hình ---
AUDIO_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "amthanh.wav")
EAR_THRESHOLD = 0.21
THRESHOLD_FRAMES = 15
COUNTER = 0
IS_PLAYING = False

def play_audio():
    global IS_PLAYING
    IS_PLAYING = True
    try:
        winsound.PlaySound(AUDIO_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except: pass
    threading.Event().wait(2)
    IS_PLAYING = False

def calculate_ear(landmarks, eye_indices):
    v1 = math.dist([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y], 
                   [landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    v2 = math.dist([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y], 
                   [landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    h = math.dist([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y], 
                  [landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    return (v1 + v2) / (2.0 * h)

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.VIDEO)

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    print("Hệ thống đã sẵn sàng. Nhấn 'q' để thoát.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            
            ear = (calculate_ear(landmarks, LEFT_EYE) + calculate_ear(landmarks, RIGHT_EYE)) / 2.0
            
            # --- LOGIC GIAO DIỆN THÔNG MINH ---
            if ear < EAR_THRESHOLD:
                COUNTER += 1
                # 1. Vẽ khung mắt màu đỏ khi có dấu hiệu buồn ngủ
                for i in LEFT_EYE + RIGHT_EYE:
                    pt = landmarks[i]
                    cv2.circle(frame, (int(pt.x * w_frame), int(pt.y * h_frame)), 2, (0, 0, 255), -1)
                
                # 2. Vẽ thanh tiến trình màu vàng
                bar_width = int((COUNTER / THRESHOLD_FRAMES) * 200)
                cv2.rectangle(frame, (50, 100), (50 + bar_width, 120), (0, 255, 255), -1)
                
                # 3. Thông báo cảnh báo
                if COUNTER >= THRESHOLD_FRAMES:
                    cv2.putText(frame, "CANH BAO NGU GAT!", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not IS_PLAYING:
                        threading.Thread(target=play_audio, daemon=True).start()
            else:
                COUNTER = 0

        cv2.imshow('Drowsiness Detection - Dai IT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()