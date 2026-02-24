import os
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
ball_model = YOLO("yolov8n.pt")
BALL_CLASS_ID = 32  # 공 클래스 ID

# 입력/출력 경로 설정
input_bases = {
    "keypoints_csv": "keypoints_csv_with_ball",
    "keypoints_csv/foot": "keypoints_foot_with_ball"
}

def add_ball_to_npy():
    for input_base, output_base in input_bases.items():
        os.makedirs(output_base, exist_ok=True)

        for label in os.listdir(input_base):
            input_folder = os.path.join(input_base, label)
            output_folder = os.path.join(output_base, label)
            os.makedirs(output_folder, exist_ok=True)

            if not os.path.exists(input_folder):
                continue

            for file in os.listdir(input_folder):
                if not file.endswith(".npy"):
                    continue

                video_name = file.replace(".npy", ".mp4")
                if "foot" in input_base:
                    video_path = os.path.join("data/foot_videos", label, video_name)
                else:
                    video_path = os.path.join("data/videos", label, video_name)

                if not os.path.exists(video_path):
                    print(f"영상 파일 없음: {video_path}")
                    continue

                npy_path = os.path.join(input_folder, file)
                data = np.load(npy_path)

                cap = cv2.VideoCapture(video_path)
                if data.ndim != 3 or data.shape[1] != 33:
                    print(f"형식 오류로 스킵됨: {file}")
                    continue

                foot = data[:, 32, :2]
                cz = 0.0
                frame_idx = 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                all_frames = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_idx >= len(data):
                        break
                    all_frames.append(frame)
                    frame_idx += 1
                cap.release()

                # 첫 번째 패스: 공 중심 추정
                ball_centers = []
                for frame in all_frames:
                    results = ball_model.predict(frame, verbose=False)
                    cx, cy = None, None
                    for r in results:
                        for box in r.boxes:
                            if int(box.cls[0]) == BALL_CLASS_ID:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cx = (x1 + x2) / 2 / width
                                cy = (y1 + y2) / 2 / height
                                break
                    if cx is not None and cy is not None:
                        ball_centers.append([cx, cy])
                    else:
                        ball_centers.append([np.nan, np.nan])

                ball_centers = np.array(ball_centers)
                distances = np.linalg.norm(ball_centers - foot, axis=1)
                if np.all(np.isnan(distances)):
                    print(f"공 검출 실패: {file}")
                    continue

                impact = np.nanargmin(distances)
                cz = data[impact, 32, 2]

                # 두 번째 패스: 공 좌표 및 cz 추가
                frames_with_ball = []
                for i, frame in enumerate(all_frames):
                    results = ball_model.predict(frame, verbose=False)
                    ball_coord = np.zeros((1, 4))
                    for r in results:
                        for box in r.boxes:
                            if int(box.cls[0]) == BALL_CLASS_ID:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cx = (x1 + x2) / 2 / width
                                cy = (y1 + y2) / 2 / height
                                ball_coord = np.array([[cx, cy, cz, 1.0]])
                                break

                    if data.shape[1] < 101:
                        pad_len = 101 - data.shape[1]
                        padding = np.zeros((pad_len, 4))
                        frame_full = np.concatenate([data[i], padding], axis=0)
                        frame_full[100] = ball_coord
                    else:
                        frame_full = np.copy(data[i])
                        frame_full[100] = ball_coord

                    frames_with_ball.append(frame_full)

                result = np.stack(frames_with_ball)
                out_path = os.path.join(output_folder, file)
                np.save(out_path, result)
                print(f"저장 완료: {out_path} ({result.shape})")

if __name__ == "__main__":
    add_ball_to_npy()
