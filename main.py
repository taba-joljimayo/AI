import eventlet
eventlet.monkey_patch()  # eventlet을 올바르게 설정



import cv2
import dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
from flask import Flask
from flask_socketio import SocketIO, emit

# Flask 애플리케이션 초기화
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# 설정
IMG_SIZE = (34, 26)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = load_model('2024_12_06_02_25_04.keras', compile=True)


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


@socketio.on('process_image')
def handle_binary_image(data):
    try:
        # 수신 데이터 디코딩
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            emit('result', 'Error: Failed to decode image data')
            return

        # 그레이스케일 변환 및 처리
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            emit('result', 'No face detected')
            return

        emit('result', 'Image processed successfully')
        results = []
        for face in faces:
            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            # 얼굴 정렬
            aligned_img = align_face(img, shapes)
            aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)

            # 정렬된 얼굴에서 랜드마크 다시 감지
            aligned_shapes = predictor(aligned_gray, face)
            aligned_shapes = face_utils.shape_to_np(aligned_shapes)

            # 왼쪽 및 오른쪽 눈 추출
            eye_img_l, eye_rect_l = crop_eye(aligned_gray, eye_points=aligned_shapes[36:42])
            eye_img_r, eye_rect_r = crop_eye(aligned_gray, eye_points=aligned_shapes[42:48])

            eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)

            # 모델 예측
            eye_input_l = eye_img_l.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
            eye_input_r = eye_img_r.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

            pred_l = model.predict(eye_input_l)[0][0]
            pred_r = model.predict(eye_input_r)[0][0]

            # 눈 상태 결과 생성
            state_l = '1' if pred_l > 0.5 else '0'
            state_r = '1' if pred_r > 0.5 else '0'

            results.append(f"Left: {state_l}, Right: {state_r}")

        # 문자열 결과 반환
        emit('result', '; '.join(results))
    except Exception as e:
        emit('result', f"Error: {str(e)}")


@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, log_output=True)
