import cv2
import mediapipe as mp
import winsound
import threading
import os
import math

# --- Cấu hình file ---
dir_path = os.path.dirname(os.path.realpath(__file__))
AUDIO_PATH = os.path.join(dir_path, "amthanh.wav")

# --- Ngưỡng EAR (Tỉ lệ chuẩn) ---
# Thường EAR < 0.2 là nhắm mắt. 
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

# Hàm tính EAR (Eye Aspect Ratio)
def calculate_ear(landmarks, left_indices):
    # Khoảng cách dọc
    v1 = math.dist([landmarks[left_indices[1]].x, landmarks[left_indices[1]].y], 
                   [landmarks[left_indices[5]].x, landmarks[left_indices[5]].y])
    v2 = math.dist([landmarks[left_indices[2]].x, landmarks[left_indices[2]].y], 
                   [landmarks[left_indices[4]].x, landmarks[left_indices[4]].y])
    # Khoảng cách ngang
    h = math.dist([landmarks[left_indices[0]].x, landmarks[left_indices[0]].y], 
                  [landmarks[left_indices[3]].x, landmarks[left_indices[3]].y])
    return (v1 + v2) / (2.0 * h)

# Cấu hình MediaPipe
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.VIDEO)

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            
            # Chỉ số các điểm mốc cho mắt trái (MediaPipe)
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            ear = calculate_ear(landmarks, LEFT_EYE)
            
            # Debug: Hiển thị EAR lên màn hình để Đại căn chỉnh
           # cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= THRESHOLD_FRAMES:
                    cv2.putText(frame, "CANH BAO NGU GAT!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not IS_PLAYING:
                        threading.Thread(target=play_audio, daemon=True).start()
            else:
                COUNTER = 0

        cv2.imshow('Drowsiness Detection - Dai IT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()