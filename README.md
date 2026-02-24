# ⚽ Lightweight Soccer Kick Analysis

BlazePose + YOLOv8 + LSTM 기반의 경량 축구 킥 분석 시스템  
(동적 양자화 적용)

---

## 📌 Overview

본 프로젝트는 소비자용 카메라 영상에서

1. BlazePose로 3D Pose 추출
2. YOLOv8로 축구공 좌표 추출
3. 임팩트 프레임 자동 추정
4. LSTM 기반 무게중심 / 디딤발 상태 분류
5. Tkinter GUI 기반 실시간 피드백 제공

을 수행하는 경량 AI 시스템입니다.

---

## 🏗 Architecture

Video Input  
→ Pose Extraction (BlazePose)  
→ Ball Detection (YOLOv8)  
→ Impact Frame Detection  
→ LSTM Classification  
→ GUI Feedback

---

## 📂 Repository Structure

data_processing/ # Pose & Ball 데이터 전처리
training/ # LSTM 학습 및 실험 코드
gui/ # GUI 실행 코드
docs/ # 실험 결과 그래프


---

## ⚙️ Tech Stack

- PyTorch
- LSTM
- Model Quantization (Dynamic Quantization)
- OpenCV
- MediaPipe (BlazePose)
- YOLOv8
- Tkinter

---

## 🚀 Key Features

- 단일 프레임 기반 무게중심 분류 실험
- 시퀀스 기반 LSTM 비교 실험
- FP32 vs Quantized 모델 성능 비교
- 경량 모델 설계 및 실시간 GUI 적용

---

## 📊 Experimental Results

실험 결과 그래프는 `docs/` 폴더 참조.

---

## 📝 Note

- Raw dataset 및 학습된 weights는 포함하지 않음.
- 연구 및 포트폴리오 목적의 프로젝트입니다.
