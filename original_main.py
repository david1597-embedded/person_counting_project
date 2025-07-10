from pathlib import Path
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter
import cv2
import time
import collections
import numpy as np
from IPython import display
import torch
import openvino as ov

from download import download

det_model, det_model_path = download()

def run_inference(source, device):
    core = ov.Core()

    det_ov_model = core.read_model(det_model_path)
    ov_config = {}

 
    compiled_model = core.compile_model(det_ov_model, 'CPU', ov_config)

    def infer(*args):
        result = compiled_model(args)
        return torch.from_numpy(result[0])

    # Use openVINO as inference engine
    det_model.predictor.inference = infer
    det_model.predictor.model.pt = False

    try:
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"

        line_points = [(0, 300), (1080, 300)]  # line or region points

        # Init Object Counter
        counter = ObjectCounter(show=False, region=line_points, model=det_model_path.parent, line_width=2, show_in=False, show_out=False)
        compiled_model.track = counter.model.track
        counter.model = compiled_model
        # Processing time
        processing_times = collections.deque(maxlen=200)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            start_time = time.time()
            frame = counter.count(frame)
            stop_time = time.time()

            processing_times.append(stop_time - start_time)

            # Mean processing time [ms].
            _, f_width = frame.shape[:2]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Get the counts. Counts are getting as 'OUT'
            # Modify this logic accordingly
            counts = counter.out_count

            # Define the text to display
            text = f"Count: {counts}"
            fontFace = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 0.75  # Adjust scale as needed
            thickness = 2

            # Calculate the size of the text box
            (text_width, text_height), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)

            # Define the upper right corner for the text
            top_right_corner = (frame.shape[1] - text_width - 20, 40)
            # Draw the count of "OUT" on the frame
            cv2.putText(
                img=frame,
                text=text,
                org=(top_right_corner[0], top_right_corner[1]),
                fontFace=fontFace,
                fontScale=fontScale,
                color=(0, 0, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.imshow('frame', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted")

    cap.release()
    cv2.destroyAllWindows()


from notebook_utils import download_file

WEBCAM_INFERENCE = True  # True : Webcam / False : .mp4

VIDEO_SOURCE = None
if WEBCAM_INFERENCE:
    VIDEO_SOURCE = 0  # Webcam
else:
    VIDEO_SOURCE = Path("./people-detection.mp4")
    if not VIDEO_SOURCE.exists():
        download_file("https://storage.openvinotoolkit.org/data/test_data/videos/people-detection.mp4")
print(VIDEO_SOURCE)
run_inference(
    source=VIDEO_SOURCE,
    device = 'CPU',
)

