import cv2
import dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import time

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 모델을 compile=False로 로드
model = load_model('2024_12_06_02_25_04.keras', compile=True)
model.summary()

def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)
    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect

def align_face(img, shapes):
    """
    얼굴 정렬(기울기 보정)을 수행합니다.
    """
    left_eye_center = np.mean(shapes[36:42], axis=0)
    right_eye_center = np.mean(shapes[42:48], axis=0)

    # 두 눈 사이의 기울기를 계산
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # 중심점과 각도를 기반으로 이미지를 회전
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                   (left_eye_center[1] + right_eye_center[1]) // 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return aligned_img

# 웹캠으로 변경
cap = cv2.VideoCapture(0)  # 0번 디바이스 (기본 웹캠)

# 감지 간격 설정 (초)
detect_interval = 0.1  # 0.3초마다 감지
last_detect_time = time.time()

while cap.isOpened():
    ret, img_ori = cap.read()

    if not ret:
        break

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 0.3초마다 얼굴 감지 실행
    if time.time() - last_detect_time > detect_interval:
        faces = detector(gray)
        last_detect_time = time.time()
    else:
        faces = []  # 감지 대기 상태

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        # 얼굴 정렬
        aligned_img = align_face(img, shapes)
        aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

        # 정렬된 얼굴에서 랜드마크 다시 감지
        aligned_shapes = predictor(aligned_gray, face)
        aligned_shapes = face_utils.shape_to_np(aligned_shapes)

        # 왼쪽과 오른쪽 눈 추출
        eye_img_l, eye_rect_l = crop_eye(aligned_gray, eye_points=aligned_shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(aligned_gray, eye_points=aligned_shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        cv2.imshow('Left Eye', eye_img_l)
        cv2.imshow('Right Eye', eye_img_r)

        # 눈 상태 예측
        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)

        # 상태 표시
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l.item()
        state_r = state_r % pred_r.item()
        # 결과 시각화
        cv2.rectangle(aligned_img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
        cv2.rectangle(aligned_img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

        cv2.putText(aligned_img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(aligned_img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Aligned Result', aligned_img)

    # 원본 영상 표시
    cv2.imshow('Original Frame', img)

    if cv2.waitKey(1) == ord('q'):  # 'q'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
