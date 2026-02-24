import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp

# MediaPipe BlazePose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 입력 폴더
kick_video_root = "data/videos"
foot_video_root = "data/foot_videos"

# 출력 폴더
csv_root = "keypoints_csv"
video_root = "output_videos"

# 디딤발 라벨
foot_labels = ["aligned", "too_close", "too_far", "in_front", "behind"]

def process_general_kick_videos():
    for label in os.listdir(kick_video_root):
        input_folder = os.path.join(kick_video_root, label)
        if not os.path.isdir(input_folder):
            continue

        csv_folder = os.path.join(csv_root, label)
        video_out_folder = video_root
        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(video_out_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if not filename.endswith('.mp4'):
                continue

            video_path = os.path.join(input_folder, filename)
            print(f"[Kick] Processing: {label}/{filename}")
            process_video(video_path, csv_folder, os.path.join(video_out_folder, f"{label}_{filename}"))

def process_foot_videos():
    for label in foot_labels:
        input_folder = os.path.join(foot_video_root, label)
        if not os.path.isdir(input_folder):
            continue

        csv_folder = os.path.join(csv_root, "foot", label)
        video_out_folder = os.path.join(video_root, f"foot_{label}")
        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(video_out_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if not filename.endswith('.mp4'):
                continue

            video_path = os.path.join(input_folder, filename)
            print(f"[Foot] Processing: {label}/{filename}")
            process_video(video_path, csv_folder, os.path.join(video_out_folder, filename))

def process_video(video_path, csv_folder, out_video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    keypoints_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image)

        if result.pose_landmarks:
            for idx, lm in enumerate(result.pose_landmarks.landmark):
                keypoints_data.append({
                    'frame': frame_idx,
                    'landmark_index': idx,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        out_video.write(frame)
        frame_idx += 1

    cap.release()
    out_video.release()

    # CSV 저장
    filename = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(csv_folder, f"{filename}.csv")
    df = pd.DataFrame(keypoints_data)
    df.to_csv(csv_path, index=False)
    print(f"CSV 저장 완료: {csv_path}")

    # NPY 저장
    if not df.empty:
        df_grouped = df.groupby("frame").apply(lambda x: x[['x', 'y', 'z', 'visibility']].values).values
        pose3d_array = np.stack(df_grouped)
        npy_path = csv_path.replace('.csv', '.npy')
        np.save(npy_path, pose3d_array)
        print(f"NPY 저장 완료: {npy_path}")

    print(f"비디오 저장 완료: {out_video_path}\n")

if __name__ == "__main__":
    process_general_kick_videos()
    process_foot_videos()
