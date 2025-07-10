
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

class PersonCounter:
    def __init__(self, model_path, counting_line=[(0, 300), (1080, 300)], device='CPU'):
        self.model_path = model_path
        self.device = device
        self.counting_line = counting_line
        self.processing_times = collections.deque(maxlen=200)

        self.openvino_model_setup()
        self.counter_setup()
    
    def openvino_model_setup(self):
        self.core = ov.Core()
        self.det_ov_model = self.core.read_model(self.model_path)
        self.compiled_model = self.core.compile_model(self.det_ov_model, self.device, {})

        self.det_model, self.det_model_path = download()
        self.det_model.predictor.inference = self.infer
        self.det_model.predictor.model.pt = False

    def infer(self, *args):
        result = self.compiled_model(args)
        return torch.from_numpy(result[0])
    
    def counter_setup(self):
        self.counter = ObjectCounter(show=False, region=self.counting_line, model=self.det_model_path.parent, line_width=2, show_in=False, show_out=False)
        self.compiled_model.track = self.counter.model.track
        self.counter.model = self.compiled_model

    def video_start(self, source):
        try:
            cap = cv2.VideoCapture(source)
            assert cap.isOpened(), "Error reading video file"

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")
                    break

                start_time = time.time()
                frame = self.counter.count(frame)
                stop_time = time.time()

                self.processing_times.append(stop_time - start_time)

                # Mean processing time [ms].
                _, f_width = frame.shape[:2]
                processing_time = np.mean(self.processing_times) * 1000
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
                counts = self.counter.out_count

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

                cv2.imshow("PersonCounter", frame)

                # q 키 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted")

        cap.release()
        cv2.destroyAllWindows()