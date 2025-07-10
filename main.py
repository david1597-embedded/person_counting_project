from pathlib import Path

from PersonCounter import PersonCounter
from download import download_file

# --- 비디오 소스 설정 ---
USE_WEBCAM = False

if USE_WEBCAM:
    VIDEO_SOURCE = 0  # 기본 webcam
else:
    VIDEO_SOURCE = Path("./topView_3.webm")
    if not VIDEO_SOURCE.exists():
        download_file("https://storage.openvinotoolkit.org/data/test_data/videos/people-detection.mp4")

if __name__=="__main__":
    counter = PersonCounter(model_path="./models/yolov8n_openvino_model/yolov8n.xml")
    counter.video_start(VIDEO_SOURCE)

