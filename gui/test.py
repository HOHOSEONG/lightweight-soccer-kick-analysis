import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
import threading
import queue
import time
import psutil # 메모리 사용량
from PIL import ImageFont, ImageDraw, Image # Pillow 임포트

# --- 모델 클래스 정의 ---
class ImpactLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class FootLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# --- 설정값 ---
IMPACT_SEQUENCE_LENGTH = 6
IMPACT_NUM_LANDMARKS_POSE_ONLY = 33
IMPACT_INPUT_SIZE = IMPACT_NUM_LANDMARKS_POSE_ONLY * 3
IMPACT_HIDDEN_SIZE = 64
IMPACT_MODEL_PATH = f"models/weights/impact_model_quantized_dynamic_h{IMPACT_HIDDEN_SIZE}.pth"
IMPACT_LABELS = {0:'back (무게중심 공보다 뒤)', 1:'front (무게중심 공보다 앞)', 2:'middle (무게중심 공과 나란히)'}
IMPACT_NUM_CLASSES = len(IMPACT_LABELS)

FOOT_SEQUENCE_LENGTH = 11
FOOT_NUM_LANDMARKS_TOTAL = 101
FOOT_INPUT_SIZE = FOOT_NUM_LANDMARKS_TOTAL * 3 + 2
FOOT_HIDDEN_SIZE = 64
FOOT_MODEL_PATH = f"models/weights/foot_lstm_model_quantized_dynamic_h{FOOT_HIDDEN_SIZE}.pth"
FOOT_LABELS = {0:"aligned", 1:"behind", 2:"in_front", 3:"too_close", 4:"too_far"}
FOOT_NUM_CLASSES = len(FOOT_LABELS)

RIGHT_FOOT_INDEX_BLAZEPOSE = 32
BALL_LANDMARK_INDEX_IN_FOOT_MODEL = 100
BALL_CLASS_ID_YOLO = 32

# --- 전역 변수 ---
pose_detector = None
ball_model_yolo = None
impact_model = None
foot_model = None
device = None
kick_type_var = None
app_running = False
result_queue = queue.Queue()
current_process = psutil.Process()
selected_video_path = None
app = None
log_text_area = None
result_text_area = None
start_analysis_button = None
progress_bar = None
video_path_label = None

def get_memory_usage():
    return current_process.memory_info().rss / (1024 * 1024)

# --- GUI 로깅 함수 ---
def log_message_to_gui(message):
    global app, log_text_area, result_text_area, start_analysis_button, progress_bar
    if not app_running or not app or not log_text_area: return
    try:
        if isinstance(message, tuple):
            log_type, content = message
            if log_type == "log":
                log_text_area.insert(tk.END, content + "\n")
            elif log_type == "result":
                result_text_area.config(state=tk.NORMAL)
                result_text_area.delete(1.0, tk.END)
                result_text_area.insert(tk.END, content + "\n")
                result_text_area.config(state=tk.DISABLED)
            elif log_type == "error":
                log_text_area.insert(tk.END, f"오류: {content}\n", "error_tag")
            elif log_type == "done":
                start_analysis_button.config(text="분석 시작", state=tk.NORMAL)
                progress_bar.stop()
                progress_bar.config(value=0)
                video_path, feedback, impact_frame, impact_landmarks, impact_ball_pos = content
                if messagebox.askyesno("재생 확인", "분석된 영상에 피드백을 추가하여 재생하시겠습니까?"):
                    play_slow = messagebox.askyesno("속도 선택", "느린 속도(0.5배속)로 재생하시겠습니까?\n('아니오' 선택 시 일반 속도로 재생됩니다.)")
                    play_video_with_feedback(video_path, feedback, impact_frame, impact_landmarks, impact_ball_pos, slow_motion=play_slow)
        else:
            log_text_area.insert(tk.END, message + "\n")
        log_text_area.see(tk.END)
    except tk.TclError as e:
        print(f"Tkinter TclError in log_message_to_gui: {e}")
    except Exception as e:
        print(f"log_message_to_gui 오류: {e}")
        try:
            log_text_area.insert(tk.END, f"GUI 업데이트 오류: {e}\n", "error_tag")
        except: pass

def initialize_components():
    global pose_detector, ball_model_yolo, impact_model, foot_model, device, FOOT_NUM_CLASSES
    mem_before_init = get_memory_usage(); log_message_to_gui(f"초기화 전 메모리: {mem_before_init:.2f} MB")
    device = torch.device("cpu"); log_message_to_gui(f"Using device: {device} (Quantized models on CPU)")
    pose_detector = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    log_message_to_gui(f"✅ MediaPipe Pose 로드 (model_complexity=2, static_image_mode=True)")
    try: ball_model_yolo=YOLO("yolov8n.pt"); log_message_to_gui("✅ YOLOv8 로드 성공.")
    except Exception as e: log_message_to_gui(f"⚠️ YOLO 로드 오류: {e}."); ball_model_yolo=None
    mem_before_lstm_load = get_memory_usage(); log_message_to_gui(f"LSTM 모델 로드 전 메모리: {mem_before_lstm_load:.2f} MB")
    if os.path.exists(IMPACT_MODEL_PATH):
        try: impact_model=torch.load(IMPACT_MODEL_PATH,map_location=device); impact_model.eval(); log_message_to_gui(f"✅ 양자화 ImpactLSTM 로드: {IMPACT_MODEL_PATH}")
        except Exception as e: log_message_to_gui(f"⚠️ 양자화 Impact모델 로드 오류({IMPACT_MODEL_PATH}): {e}"); impact_model=None
    else: log_message_to_gui(f"⚠️ 양자화 Impact모델 없음: {IMPACT_MODEL_PATH}.")
    if FOOT_NUM_CLASSES != len(FOOT_LABELS) and len(FOOT_LABELS) > 0:
        log_message_to_gui(f"알림: FOOT_NUM_CLASSES({FOOT_NUM_CLASSES})를 FOOT_LABELS 길이({len(FOOT_LABELS)})로 업데이트합니다."); FOOT_NUM_CLASSES = len(FOOT_LABELS)
    if os.path.exists(FOOT_MODEL_PATH):
        try: foot_model=torch.load(FOOT_MODEL_PATH,map_location=device); foot_model.eval(); log_message_to_gui(f"✅ 양자화 FootLSTM 로드: {FOOT_MODEL_PATH}")
        except Exception as e: log_message_to_gui(f"⚠️ 양자화 Foot모델 로드 오류({FOOT_MODEL_PATH}): {e}"); foot_model=None
    else: log_message_to_gui(f"⚠️ 양자화 Foot모델 없음: {FOOT_MODEL_PATH}.")
    mem_after_init=get_memory_usage(); log_message_to_gui(f"LSTM 모델 로드 후 메모리: {mem_after_init:.2f} MB (증가분 추정: {(mem_after_init-mem_before_lstm_load):.2f} MB)")
    log_message_to_gui(f"전체 초기화 후 메모리: {mem_after_init:.2f} MB (초기화 전 대비 증가: {(mem_after_init - mem_before_init):.2f} MB)")

def generate_feedback(kick_type, impact_label_str, foot_label_str):
    fb_msgs = []
    imp_clean = "n/a"; foot_clean = "n/a"
    if impact_label_str and impact_label_str != "N/A" and not impact_label_str.startswith("Unk"): imp_clean = impact_label_str.split(" ")[0].lower()
    if foot_label_str and foot_label_str != "N/A" and not foot_label_str.startswith("Unk"): foot_clean = foot_label_str.lower()
    if kick_type == "롱 패스" and imp_clean != "back": fb_msgs.append(f"무게중심: 뒤(back) 권장 (현재: {impact_label_str})")
    elif kick_type == "짧은 패스" and imp_clean != "middle": fb_msgs.append(f"무게중심: 중앙(middle) 권장 (현재: {impact_label_str})")
    elif kick_type == "슈팅" and imp_clean != "front": fb_msgs.append(f"무게중심: 앞(front) 권장 (현재: {impact_label_str})")
    else: fb_msgs.append(f"무게중심: 이상적! (현재: {impact_label_str})")
    if foot_clean != "aligned" and foot_clean != "n/a": fb_msgs.append(f"디딤발: 정렬(aligned) 권장 (현재: {foot_label_str})")
    else: fb_msgs.append(f"디딤발: 이상적! (현재: {foot_label_str})")
    return "\n".join(fb_msgs)

def draw_text_with_bg(draw, pos_px, text, font, text_color=(255, 255, 0, 255), bg_color=(0, 0, 0, 128)):
    if not text or pos_px is None: return
    try:
        text_bbox = draw.textbbox(pos_px, text, font=font)
        bg_bbox = (text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5)
        draw.rectangle(bg_bbox, fill=bg_color)
        draw.text(pos_px, text, font=font, fill=text_color)
    except Exception as e:
        print(f"텍스트 그리기 오류: {e}, Text: {text}, Pos: {pos_px}")

# <<< 영상 재생 함수 수정: 일시정지 기능 추가 >>>
def play_video_with_feedback(video_path, feedback_text, impact_frame, impact_landmarks, impact_ball_pos, slow_motion=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("오류", f"비디오를 열 수 없습니다: {video_path}")
        return

    font_path = "malgun.ttf"
    if not os.path.exists(font_path): font_path = "C:/Windows/Fonts/malgun.ttf"
    font = None
    try: font = ImageFont.truetype(font_path, 20)
    except IOError: log_message_to_gui(f"⚠️ 폰트 파일을 찾을 수 없습니다: {font_path}.")

    fps = cap.get(cv2.CAP_PROP_FPS); delay = int(1000 / fps) if fps > 0 else 25
    speed_factor = 0.5 if slow_motion else 1.0; actual_delay = int(delay / speed_factor)
    speed_text = f"Speed: {speed_factor:.1f}x (Press 'q' to quit)"

    weight_fb = ""; foot_fb = ""
    for line in feedback_text.split('\n'):
        if "무게중심" in line: weight_fb = line
        elif "디딤발" in line: foot_fb = line

    back_pos_px = None; foot_pos_px = None
    temp_w, temp_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if impact_landmarks is not None:
        lm11, lm12 = impact_landmarks[11], impact_landmarks[12]
        if 0 < lm11[0] < 1 and 0 < lm12[0] < 1:
            bx = (lm11[0] + lm12[0]) / 2; by = (lm11[1] + lm12[1]) / 2 - 0.1
            back_pos_px = (int(bx * temp_w), int(by * temp_h))

    if impact_ball_pos is not None:
        bx, by = impact_ball_pos[0], impact_ball_pos[1]
        if 0 < bx < 1 and 0 < by < 1:
            foot_pos_px = (int(bx * temp_w), int(by * temp_h + 30))
    elif impact_landmarks is not None:
        lm32 = impact_landmarks[32]
        if 0 < lm32[0] < 1:
            foot_pos_px = (int(lm32[0] * temp_w), int(lm32[1] * temp_h + 30))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(img_pil)

        if font:
            draw_text_with_bg(draw, back_pos_px, weight_fb, font, (255, 255, 0, 255))
            draw_text_with_bg(draw, foot_pos_px, foot_fb, font, (0, 255, 255, 255))

        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)

        if impact_frame - 2 <= frame_num <= impact_frame + 2:
             cv2.putText(frame, "IMPACT", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        (text_width, text_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, speed_text, (w - text_width - 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        # <<< 일시 정지 로직 추가 >>>
        if frame_num == impact_frame:
            pause_text = "PAUSED - Press any key to continue"
            (pw, ph), _ = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, pause_text, (w // 2 - pw // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Feedback Video", frame)
            key = cv2.waitKey(0) # 여기서 일시 정지
            if key & 0xFF == ord('q'): # 'q' 누르면 종료
                break
        else:
            cv2.imshow("Feedback Video", frame)
            key = cv2.waitKey(actual_delay) & 0xFF
            if key == ord('q'): # 'q' 누르면 종료
                break
        # <<< 일시 정지 로직 끝 >>>

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


def analyze_video_for_single_kick_event_threaded(video_path, selected_kick_type, msg_queue):
    mem_before_analysis = get_memory_usage(); msg_queue.put(("log", f"분석 시작 전 메모리: {mem_before_analysis:.2f} MB"))
    overall_start_t = time.perf_counter()
    if not video_path: msg_queue.put(("error", "비디오 미선택")); return
    if not impact_model or not foot_model:
        missing = [m for m, loaded in [("Impact", impact_model), ("Foot", foot_model)] if not loaded]
        msg_queue.put(("error", f"필수 모델 ({','.join(missing)}) 로드 안됨")); return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): msg_queue.put(("error", f"비디오 열기 오류: {video_path}")); return

    s1_d_s, imp_est_t_s, imp_inf_t_ms, foot_inf_t_ms, frame_proc_t_s = 0, 0, 0, 0, 0
    s1_start_ns = time.perf_counter_ns(); raw_frames = []
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_f == 0: msg_queue.put(("log", f"'{os.path.basename(video_path)}' 프레임 정보 없음")); cap.release(); return
    msg_queue.put(("log", f"\n'{os.path.basename(video_path)}' ({selected_kick_type}) 1단계: 프레임 추출 (총 {total_f}프레임)"))
    proc_f_cnt = 0; single_f_proc_ns_sum = 0
    while cap.isOpened():
        ret, frm = cap.read()
        if not ret: break
        sf_start_ns = time.perf_counter_ns(); proc_f_cnt += 1
        if proc_f_cnt % 60 == 0: msg_queue.put(("log", f"  {proc_f_cnt}/{total_f} 처리 중..."))
        img_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        pose_res_mp = pose_detector.process(img_rgb)
        current_pose_landmarks_33 = np.zeros((IMPACT_NUM_LANDMARKS_POSE_ONLY, 3), dtype=np.float32)
        if pose_res_mp.pose_landmarks:
            for i, lm in enumerate(pose_res_mp.pose_landmarks.landmark):
                if i < IMPACT_NUM_LANDMARKS_POSE_ONLY: current_pose_landmarks_33[i] = [lm.x, lm.y, lm.z]
        ball_xyz_raw = None
        if ball_model_yolo:
            yolo_res = ball_model_yolo.predict(frm, verbose=False, classes=[BALL_CLASS_ID_YOLO])
            if yolo_res and yolo_res[0].boxes:
                for box in yolo_res[0].boxes:
                    if int(box.cls[0]) == BALL_CLASS_ID_YOLO:
                        x1, y1, x2, y2 = map(int, box.xyxy[0]); cxn = (x1 + x2) / (2 * w); cyn = (y1 + y2) / (2 * h); t_bz = 0.5
                        if pose_res_mp.pose_landmarks and RIGHT_FOOT_INDEX_BLAZEPOSE < len(pose_res_mp.pose_landmarks.landmark) and np.any(current_pose_landmarks_33[RIGHT_FOOT_INDEX_BLAZEPOSE]):
                            t_bz = current_pose_landmarks_33[RIGHT_FOOT_INDEX_BLAZEPOSE, 2]
                        ball_xyz_raw = np.array([cxn, cyn, t_bz], dtype=np.float32); break
        raw_frames.append({'pose': current_pose_landmarks_33, 'ball_raw': ball_xyz_raw})
        single_f_proc_ns_sum += (time.perf_counter_ns() - sf_start_ns)
    cap.release(); s1_end_ns = time.perf_counter_ns()
    s1_d_s = (s1_end_ns - s1_start_ns) / 1e9; frame_proc_t_s = single_f_proc_ns_sum / 1e9
    msg_queue.put(("log", f"1단계 완료 (소요:{s1_d_s:.2f}s, 순수프레임처리:{frame_proc_t_s:.2f}s)."))
    if not raw_frames: msg_queue.put(("error", "데이터 추출 실패.")); return
    s2_start_ns = time.perf_counter_ns(); msg_queue.put(("log", f"\n2단계: 임팩트 추정 및 정제..."))
    f_pos, b_pos, val_idx = [], [], []
    for i, data_raw in enumerate(raw_frames):
        if data_raw['ball_raw'] is not None and np.any(data_raw['pose'][RIGHT_FOOT_INDEX_BLAZEPOSE]):
            f_pos.append(data_raw['pose'][RIGHT_FOOT_INDEX_BLAZEPOSE, :3])
            b_pos.append(data_raw['ball_raw'][:3]); val_idx.append(i)
    if not val_idx: msg_queue.put(("error", "임팩트 추정 유효 프레임 부족.")); return
    dists = np.linalg.norm(np.array(f_pos) - np.array(b_pos), axis=1)
    if len(dists) == 0: msg_queue.put(("error", "발-공 거리 계산 불가.")); return
    est_imp_idx = val_idx[np.argmin(dists)]; msg_queue.put(("log", f"추정 임팩트 프레임: {est_imp_idx}"))
    fixed_b_z = 0.5
    if np.any(raw_frames[est_imp_idx]['pose'][RIGHT_FOOT_INDEX_BLAZEPOSE]): fixed_b_z = raw_frames[est_imp_idx]['pose'][RIGHT_FOOT_INDEX_BLAZEPOSE, 2]
    all_frames_final = []
    for dr in raw_frames:
        final_ball = None
        if dr['ball_raw'] is not None: final_ball = np.array([dr['ball_raw'][0], dr['ball_raw'][1], fixed_b_z], dtype=np.float32)
        all_frames_final.append({'pose': dr['pose'], 'ball': final_ball})
    impact_frame_landmarks = all_frames_final[est_imp_idx]['pose'] if est_imp_idx < len(all_frames_final) else None
    impact_ball_pos = all_frames_final[est_imp_idx]['ball'] if est_imp_idx < len(all_frames_final) else None

    s2_end_ns = time.perf_counter_ns(); imp_est_t_s = (s2_end_ns - s2_start_ns) / 1e9
    msg_queue.put(("log", f"2단계 완료 (소요:{imp_est_t_s:.2f}s)."))
    msg_queue.put(("log", f"\n3단계: 모델 추론...")); imp_pred_str, foot_pred_str = "N/A", "N/A"
    if impact_model:
        s_imp, e_imp = max(0, est_imp_idx - 2), max(0, est_imp_idx - 2) + IMPACT_SEQUENCE_LENGTH
        if e_imp <= len(all_frames_final):
            imp_seq = [all_frames_final[i]['pose'].flatten() for i in range(s_imp, e_imp)]
            imp_tensor = torch.tensor(np.array(imp_seq), dtype=torch.float32).unsqueeze(0).to(device)
            inf_s_ns = time.perf_counter_ns()
            with torch.no_grad(): pred_i = torch.argmax(impact_model(imp_tensor), dim=1).item()
            imp_inf_t_ms = (time.perf_counter_ns() - inf_s_ns) / 1e6; imp_pred_str = IMPACT_LABELS.get(pred_i, f"Unknown_idx_{pred_i}")
        else: msg_queue.put(("log", "ImpactLSTM: 시퀀스 부족."))
    if foot_model:
        s_foot, e_foot = est_imp_idx - (FOOT_SEQUENCE_LENGTH // 2), est_imp_idx - (FOOT_SEQUENCE_LENGTH // 2) + FOOT_SEQUENCE_LENGTH
        if s_foot >= 0 and e_foot <= len(all_frames_final):
            foot_seq_final = []; feat_imp_val = np.array([0., 0.], dtype=np.float32)
            imp_f_pose_feat = all_frames_final[est_imp_idx]['pose']; imp_f_ball_feat = all_frames_final[est_imp_idx]['ball']
            if imp_f_ball_feat is not None and np.any(imp_f_pose_feat[RIGHT_FOOT_INDEX_BLAZEPOSE]):
                rf_xyz_imp = imp_f_pose_feat[RIGHT_FOOT_INDEX_BLAZEPOSE, :3]; ball_xyz_imp = imp_f_ball_feat[:3]
                dist = np.linalg.norm(rf_xyz_imp - ball_xyz_imp); dy = rf_xyz_imp[1] - ball_xyz_imp[1]; feat_imp_val = np.array([dist, dy], dtype=np.float32)
            feat_tiled = np.tile(feat_imp_val, (FOOT_SEQUENCE_LENGTH, 1))
            for i in range(s_foot, e_foot):
                frm_data = all_frames_final[i]; pose33 = frm_data['pose']; ball_curr = frm_data['ball']
                lm101 = np.zeros((FOOT_NUM_LANDMARKS_TOTAL, 3), dtype=np.float32); lm101[:IMPACT_NUM_LANDMARKS_POSE_ONLY, :] = pose33
                if ball_curr is not None: lm101[BALL_LANDMARK_INDEX_IN_FOOT_MODEL, :] = ball_curr
                lms_flat = lm101.flatten(); feat_curr = feat_tiled[i - s_foot]; foot_seq_final.append(np.concatenate([lms_flat, feat_curr]))
            if len(foot_seq_final) == FOOT_SEQUENCE_LENGTH:
                foot_tensor = torch.tensor(np.array(foot_seq_final), dtype=torch.float32).unsqueeze(0).to(device)
                inf_s_ns = time.perf_counter_ns()
                with torch.no_grad(): pred_f = torch.argmax(foot_model(foot_tensor), dim=1).item()
                foot_inf_t_ms = (time.perf_counter_ns() - inf_s_ns) / 1e6; foot_pred_str = FOOT_LABELS.get(pred_f, f"Unknown_idx_{pred_f}")
            else: msg_queue.put(("log", "FootLSTM: 시퀀스 길이 불일치."))
        else: msg_queue.put(("log", f"FootLSTM: 시퀀스 부족."))
    overall_end_s = time.perf_counter(); total_t_s = overall_end_s - overall_start_t
    mem_after_analysis = get_memory_usage(); msg_queue.put(("log", f"분석 종료 후 메모리: {mem_after_analysis:.2f} MB (변화량: {(mem_after_analysis - mem_before_analysis):.2f} MB)"))
    fbk = generate_feedback(selected_kick_type, imp_pred_str, foot_pred_str)
    res_txt = f"--- 분석 결과 ({selected_kick_type}) ---\n비디오: {os.path.basename(video_path)}\n임팩트 프레임: {est_imp_idx}\n"
    res_txt += f"무게중심 (H{IMPACT_HIDDEN_SIZE}): {imp_pred_str}\n디딤발 (H{FOOT_HIDDEN_SIZE}): {foot_pred_str}\n"
    res_txt += f"\n--- 피드백 ---\n{fbk}\n\n--- 처리 시간 ---\n총 분석: {total_t_s:.2f}s\n"
    res_txt += f"  - 프레임 추출(YOLO+Pose): {s1_d_s:.2f}s (프레임당 평균: {(frame_proc_t_s / proc_f_cnt * 1000) if proc_f_cnt > 0 else 0:.2f}ms)\n"
    res_txt += f"  - 임팩트 추정/정제: {imp_est_t_s:.2f}s\n  - ImpactLSTM 추론: {imp_inf_t_ms:.2f}ms\n  - FootLSTM 추론: {foot_inf_t_ms:.2f}ms\n"
    res_txt += f"\n--- 메모리 사용량 ---\n분석 전: {mem_before_analysis:.2f} MB\n분석 후: {mem_after_analysis:.2f} MB\n(참고: 프로세스 전체 메모리)\n"
    res_txt += "---------------------\n"; msg_queue.put(("result", res_txt))
    msg_queue.put(("done", (video_path, fbk, est_imp_idx, impact_frame_landmarks, impact_ball_pos)))

def check_queue():
    global app
    if not app_running: return
    try: message = result_queue.get_nowait(); log_message_to_gui(message)
    except queue.Empty: pass
    except NameError: pass
    if app: app.after(100, check_queue)

def select_video_file():
    global selected_video_path, video_path_label, start_analysis_button
    initial_dir = os.path.join(os.getcwd(), "data", "input_videos")
    if not os.path.exists(initial_dir): initial_dir = os.getcwd()
    filepath = filedialog.askopenfilename(initialdir=initial_dir, title="비디오 파일 선택", filetypes=(("MP4", "*.mp4"), ("AVI", "*.avi"), ("All", "*.*")))
    if filepath:
        selected_video_path = filepath; video_path_label.config(text=os.path.basename(filepath))
        log_message_to_gui(f"선택된 비디오: {filepath}"); start_analysis_button.config(state=tk.NORMAL)

def start_analysis_threaded():
    global selected_video_path, kick_type_var, result_text_area, start_analysis_button, progress_bar
    if not selected_video_path: messagebox.showerror("오류", "비디오 파일을 먼저 선택해주세요."); return
    sel_kick = kick_type_var.get();
    if not sel_kick: messagebox.showerror("입력 오류", "킥 종류를 선택해주세요."); return
    if not (impact_model and foot_model): messagebox.showerror("오류", "필수 모델이 모두 로드되지 않았습니다."); return
    log_message_to_gui(f"분석 시작: {selected_video_path} (킥: {sel_kick})")
    result_text_area.config(state=tk.NORMAL); result_text_area.delete(1.0, tk.END); result_text_area.insert(tk.END, "분석 중...\n"); result_text_area.config(state=tk.DISABLED)
    start_analysis_button.config(text="분석 중...", state=tk.DISABLED); progress_bar.start(10)
    threading.Thread(target=analyze_video_for_single_kick_event_threaded, args=(selected_video_path, sel_kick, result_queue), daemon=True).start()

def on_close():
    global app_running, app, pose_detector
    if messagebox.askokcancel("종료", "종료하시겠습니까?"):
        app_running = False
        if pose_detector:
            try: pose_detector.close()
            except Exception as e: print(f"MediaPipe 종료 오류: {e}")
        if app:
            try: app.destroy()
            except Exception as e: print(f"Tkinter 종료 오류: {e}")
        app = None

if __name__ == '__main__':
    app = tk.Tk(); app.title("축구 킥 분석기"); app.geometry("750x750"); style = ttk.Style(app); style.theme_use('clam')
    top_f = ttk.Frame(app, padding="10"); top_f.pack(fill=tk.X, pady=(0, 5)); kt_f = ttk.Frame(app, padding="10 0 10 10"); kt_f.pack(fill=tk.X, pady=(0, 5))
    mid_f = ttk.Frame(app, padding="10"); mid_f.pack(fill=tk.BOTH, expand=True); bot_f = ttk.Frame(app, padding="10"); bot_f.pack(fill=tk.X)
    ttk.Button(top_f, text="비디오 선택", command=select_video_file).pack(side=tk.LEFT, padx=5)
    video_path_label = ttk.Label(top_f, text="선택된 파일 없음", width=50, anchor="w"); video_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    start_analysis_button = ttk.Button(top_f, text="분석 시작", command=start_analysis_threaded, state=tk.DISABLED); start_analysis_button.pack(side=tk.LEFT, padx=5)
    ttk.Label(kt_f, text="킥 종류:").pack(side=tk.LEFT, padx=(0, 10)); kick_type_var = tk.StringVar()
    k_types = ["롱 패스", "짧은 패스", "슈팅"]; [ttk.Radiobutton(kt_f, text=k, variable=kick_type_var, value=k).pack(side=tk.LEFT, padx=5) for k in k_types]; kick_type_var.set(k_types[0])
    nb = ttk.Notebook(mid_f); log_f = ttk.Frame(nb, padding=5); res_f = ttk.Frame(nb, padding=5); nb.add(log_f, text='로그'); nb.add(res_f, text='결과'); nb.pack(fill=tk.BOTH, expand=True)
    log_text_area = scrolledtext.ScrolledText(log_f, wrap=tk.WORD, width=80, height=15, font=("Malgun Gothic", 9)); log_text_area.pack(fill=tk.BOTH, expand=True); log_text_area.tag_config("error_tag", foreground="red")
    result_text_area = scrolledtext.ScrolledText(res_f, wrap=tk.WORD, width=80, height=15, font=("Malgun Gothic", 10), state=tk.DISABLED); result_text_area.pack(fill=tk.BOTH, expand=True)
    progress_bar = ttk.Progressbar(bot_f, mode='indeterminate'); progress_bar.pack(fill=tk.X, pady=5)
    app_running = True; initialize_components()
    if not (impact_model and foot_model): log_message_to_gui(("error", "필수 모델 로드 실패!")); start_analysis_button.config(state=tk.DISABLED)
    check_queue(); app.protocol("WM_DELETE_WINDOW", on_close); app.mainloop()