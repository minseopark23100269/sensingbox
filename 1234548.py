import os
import cv2
import mediapipe as mp
import numpy as np
import time

# -------------------------
# Helper: 알파 채널을 고려한 이미지 오버레이 함수
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    if x >= bg_w or y >= bg_h:
        return background

    h, w = overlay.shape[:2]
    if x + w > bg_w:
        w = bg_w - x
        overlay = overlay[:, :w]
    if y + h > bg_h:
        h = bg_h - y
        overlay = overlay[:h]

    if overlay.shape[2] == 4:
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay
        for c in range(3):
            background[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                           alpha_background * background[y:y+h, x:x+w, c])
    else:
        background[y:y+h, x:x+w] = overlay
    return background

# -------------------------
# Helper: Eye Aspect Ratio (EAR) 계산 함수
def compute_EAR(landmarks, indices):
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

# -------------------------
# Helper: 상대 경로로 이미지 로드 함수
def load_image(filename):
    path = os.path.join("images", filename)
    if not os.path.exists(path):
        print(f"⚠ 파일을 찾을 수 없습니다: {path}")
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

# -------------------------
# 이미지 파일 불러오기 (상대 경로 사용)
crown_img = load_image("crown.png")
glasses_img = load_image("glasses.png")
smile_img = load_image("smile.png")
zzz_img = load_image("zzz.png")

if smile_img is not None:
    smile_img = cv2.resize(smile_img, (150, 150), interpolation=cv2.INTER_AREA)
if zzz_img is not None:
    zzz_img = cv2.resize(zzz_img, (150, 150), interpolation=cv2.INTER_AREA)

# -------------------------
# OpenCV / MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, tracking_confidence=0.8)

cap = cv2.VideoCapture(0)

# -------------------------
# 먼저 첫 프레임을 읽어 프레임 크기를 결정하고 비디오 저장 객체 생성 (상대 경로 사용)
ret, frame = cap.read()
if not ret:
    print("Webcam을 열 수 없습니다!")
    exit()
frame = cv2.flip(frame, 1)
frame_height, frame_width, _ = frame.shape
if not os.path.exists("output"):
    os.makedirs("output")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(os.path.join("output", "recorded_video.avi"), fourcc, 20.0, (frame_width, frame_height))

# -------------------------
# 트리거 원 위치 설정 (각각 충분한 간격 유지)
red_circle_pos = (100, 20)      # 빨강: 안경 효과 트리거
white_circle_pos = (200, 20)    # 흰색: 왕관 효과 트리거
yellow_circle_pos = (300, 20)   # 노랑: 카툰 효과 트리거
green_circle_pos = (400, 20)    # 초록: funny face 효과 트리거
circle_radius = 15

effect_duration = 5  # 효과 지속 시간 (초)
glasses_start_time = None
crown_start_time = None
cartoon_start_time = None
funny_face_start_time = None
smile_start_time = None
closed_eye_start_time = None

# 얼굴 관련 인덱스 (enlarge 및 EAR 계산)
left_eye_ids = [33, 7, 163, 144, 145, 153, 154, 155, 133]
right_eye_ids = [362, 382, 381, 380, 374, 373, 263, 249, 390]
mouth_ids = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# -------------------------
# 메인 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # (B) 손 검지 및 트리거 업데이트
    results_hands = hands.process(rgb_frame)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x = int(index_tip.x * frame_width)
            index_y = int(index_tip.y * frame_height)
            if np.linalg.norm(np.array([index_x, index_y]) - np.array(red_circle_pos)) < circle_radius * 1.5:
                glasses_start_time = time.time()
            if np.linalg.norm(np.array([index_x, index_y]) - np.array(white_circle_pos)) < circle_radius:
                crown_start_time = time.time()
            if np.linalg.norm(np.array([index_x, index_y]) - np.array(yellow_circle_pos)) < circle_radius:
                cartoon_start_time = time.time()
            if np.linalg.norm(np.array([index_x, index_y]) - np.array(green_circle_pos)) < circle_radius:
                funny_face_start_time = time.time()
    
    # (C) 얼굴 표정 감지 – 큰 정사각형 영역에서 Face Mesh 적용
    square_side_full = min(frame_width, frame_height)
    full_crop_x = (frame_width - square_side_full) // 2
    full_crop_y = (frame_height - square_side_full) // 2
    results_face_full = face_mesh.process(rgb_frame[full_crop_y:full_crop_y+square_side_full, 
                                                      full_crop_x:full_crop_x+square_side_full])
    if results_face_full.multi_face_landmarks:
        face_landmarks_full = results_face_full.multi_face_landmarks[0]
        # 웃음 감지 (입 폭 vs. 눈 사이 거리)
        left_mouth = face_landmarks_full.landmark[61]
        right_mouth = face_landmarks_full.landmark[291]
        mouth_width = np.linalg.norm(np.array([left_mouth.x, left_mouth.y]) - np.array([right_mouth.x, right_mouth.y]))
        left_eye_point = face_landmarks_full.landmark[133]
        right_eye_point = face_landmarks_full.landmark[362]
        eye_distance = np.linalg.norm(np.array([left_eye_point.x, left_eye_point.y]) - np.array([right_eye_point.x, right_eye_point.y]))
        smile_ratio = mouth_width / eye_distance if eye_distance > 0 else 0
        if smile_ratio > 1.6:
            if smile_start_time is None:
                smile_start_time = time.time()
            closed_eye_start_time = None
        else:
            smile_start_time = None

        # 눈 감음 감지 (EAR)
        left_EAR = compute_EAR(face_landmarks_full.landmark, left_eye_indices)
        right_EAR = compute_EAR(face_landmarks_full.landmark, right_eye_indices)
        avg_EAR = (left_EAR + right_EAR) / 2.0
        if avg_EAR < 0.25:
            if closed_eye_start_time is None:
                closed_eye_start_time = time.time()
            smile_start_time = None
        else:
            closed_eye_start_time = None

    current_time = time.time()
    active_effect = None
    if glasses_start_time is not None and (current_time - glasses_start_time < effect_duration):
        active_effect = "glasses"
    elif crown_start_time is not None and (current_time - crown_start_time < effect_duration):
        active_effect = "crown"
    elif cartoon_start_time is not None and (current_time - cartoon_start_time < effect_duration):
        active_effect = "cartoon"
    elif funny_face_start_time is not None and (current_time - funny_face_start_time < effect_duration):
        active_effect = "funny"

    frame_effect = frame.copy()

    # (D) 효과 적용
    if active_effect == "cartoon":
        frame_effect = cv2.stylization(frame_effect, sigma_s=150, sigma_r=0.25)
    elif active_effect == "funny":
        frame_effect = cv2.GaussianBlur(frame_effect, (7, 7), 0)
    elif active_effect == "glasses" and results_face_full.multi_face_landmarks and glasses_img is not None:
        face_landmarks = results_face_full.multi_face_landmarks[0]
        left_eye_center = (
            crop_x + int(((face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2) * square_side_full),
            crop_y + int(((face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2) * square_side_full)
        )
        right_eye_center = (
            crop_x + int(((face_landmarks.landmark[263].x + face_landmarks.landmark[362].x) / 2) * square_side_full),
            crop_y + int(((face_landmarks.landmark[263].y + face_landmarks.landmark[362].y) / 2) * square_side_full)
        )
        glasses_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                          (left_eye_center[1] + right_eye_center[1]) // 2)
        eye_distance = np.linalg.norm(np.array(left_eye_center) - np.array(right_eye_center))
        glasses_width = int(2.0 * eye_distance)
        if glasses_width <= 0:
            glasses_width = 150
        orig_h, orig_w = glasses_img.shape[:2]
        glasses_aspect = orig_w / orig_h
        glasses_height = int(glasses_width / glasses_aspect)
        glasses_x = glasses_center[0] - glasses_width // 2
        glasses_y = glasses_center[1] - glasses_height // 2
        resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
        frame_effect = overlay_transparent(frame_effect, resized_glasses, glasses_x, glasses_y)
    elif active_effect == "crown" and results_face_full.multi_face_landmarks and crown_img is not None:
        pts = np.array([[lm.x, lm.y] for lm in face_landmarks_full.landmark])
        x_mean = np.mean(pts[:, 0])
        y_min = np.min(pts[:, 1])
        crown_center_x = full_crop_x + int(x_mean * square_side_full)
        crown_y_pos = full_crop_y + int(y_min * square_side_full) - 20
        crown_width = 200
        if crown_width <= 0:
            crown_width = 200
        crown_aspect = crown_img.shape[1] / crown_img.shape[0]
        crown_height = int(crown_width / crown_aspect)
        crown_x = crown_center_x - crown_width // 2
        crown_y = crown_y_pos - crown_height
        resized_crown = cv2.resize(crown_img, (crown_width, crown_height), interpolation=cv2.INTER_AREA)
        frame_effect = overlay_transparent(frame_effect, resized_crown, crown_x, crown_y)
    elif active_effect == "rudolph" and results_face_full.multi_face_landmarks:
        for face_landmarks in results_face_full.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            nose_x_sq = int(nose_tip.x * square_side_full)
            nose_y_sq = int(nose_tip.y * square_side_full)
            nose_x = crop_x + nose_x_sq
            nose_y = crop_y + nose_y_sq
            cv2.circle(frame_effect, (nose_x, nose_y), 15, (0, 0, 255), -1)

    # (E) 독립적인 웃음 및 눈 감음 효과 적용 (좌측, 우측 중앙)
    if smile_start_time is not None and (current_time - smile_start_time < effect_duration) and smile_img is not None:
        overlay_h, overlay_w = smile_img.shape[:2]
        left_x = frame_width // 4 - overlay_w // 2
        right_x = (3 * frame_width) // 4 - overlay_w // 2
        y_offset = frame_height // 2 - overlay_h // 2
        frame_effect = overlay_transparent(frame_effect, smile_img, left_x, y_offset)
        frame_effect = overlay_transparent(frame_effect, smile_img, right_x, y_offset)
    
    if closed_eye_start_time is not None and (current_time - closed_eye_start_time < effect_duration) and zzz_img is not None:
        overlay_h, overlay_w = zzz_img.shape[:2]
        left_x = frame_width // 4 - overlay_w // 2
        right_x = (3 * frame_width) // 4 - overlay_w // 2
        y_offset = frame_height // 2 - overlay_h // 2
        frame_effect = overlay_transparent(frame_effect, zzz_img, left_x, y_offset)
        frame_effect = overlay_transparent(frame_effect, zzz_img, right_x, y_offset)
    
    # (F) 마지막에 트리거 원 다시 그리기 (항상 보이도록)
    cv2.circle(frame_effect, red_circle_pos, circle_radius, (0, 0, 255), -1)
    cv2.circle(frame_effect, white_circle_pos, circle_radius, (255, 255, 255), -1)
    cv2.circle(frame_effect, yellow_circle_pos, circle_radius, (0, 255, 255), -1)
    cv2.circle(frame_effect, green_circle_pos, circle_radius, (0, 255, 0), -1)
    
    cv2.imshow("Effects & Hand Zoom", frame_effect)
    
    # (G) 저장: 현재 프레임을 비디오에 기록
    writer.write(frame_effect)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
