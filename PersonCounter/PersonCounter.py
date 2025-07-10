
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
        # OpenVINO는 numpy만 받음, torch.Tensor → numpy (CPU)
        processed_inputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                processed_inputs.append(arg.detach().cpu().numpy())
            else:
                processed_inputs.append(arg)
        result = self.compiled_model(processed_inputs)
        # 결과가 numpy → torch로 되돌림 (CPU tensor)
        return torch.from_numpy(result[0])
    
    def counter_setup(self):
        self.counter = ObjectCounter(show=False, region=self.counting_line, model=self.det_model_path.parent, line_width=2, show_in=False, show_out=False)
        self.compiled_model.track = self.counter.model.track
        self.counter.model = self.compiled_model

    def counting_person(self):
        start_time = time.time()

        # === line cross counter ===
        # self.frame = self.counter.count(self.frame)

        # === frame exist counter ===
        results = self.det_model(self.frame)
        self.frame = results[0].plot()

        stop_time = time.time()

        self.processing_times.append(stop_time - start_time)

        # Mean processing time [ms].
        _, f_width = self.frame.shape[:2]
        processing_time = np.mean(self.processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(
            img=self.frame,
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
    
        # counts = self.counter.out_count
        counts = sum(1 for c in results[0].boxes.cls if int(c) == 0)

        total_person_area = 0
        frame_area = self.frame.shape[0] * self.frame.shape[1]

        # Define the text to display
        text = f"Count: {counts}"
        fontFace = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 0.75  # Adjust scale as needed
        thickness = 2

        # Calculate the size of the text box
        (text_width, text_height), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)

        # Define the upper right corner for the text
        top_right_corner = (self.frame.shape[1] - text_width - 20, 40)

        for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            if int(cls_id) == 0:  # class 0 = person
                counts += 1
                x1, y1, x2, y2 = map(int, box)
                area = max((x2 - x1), 0) * max((y2 - y1), 0)
                total_person_area += area

        congestion_ratio = (total_person_area / frame_area) * 100  # %

        text_count = f"Count: {counts}"
        text_congestion = f"Congestion: {congestion_ratio:.1f}%"

        # Count 표시
        (tw, th), _ = cv2.getTextSize(text_count, cv2.FONT_HERSHEY_COMPLEX, 0.75, 2)
        pos_x = self.frame.shape[1] - tw - 20
        cv2.putText(
            self.frame, text_count, (pos_x, 40),
            cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA
        )

        # Congestion 표시
        (tw2, th2), _ = cv2.getTextSize(text_congestion, cv2.FONT_HERSHEY_COMPLEX, 0.75, 2)
        pos_x2 = self.frame.shape[1] - tw2 - 20
        cv2.putText(
            self.frame, text_congestion, (pos_x2, 80),
            cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA
        )

        # Draw the count of "OUT" on the frame
        cv2.putText(
            img=self.frame,
            text=text,
            org=(top_right_corner[0], top_right_corner[1]),
            fontFace=fontFace,
            fontScale=fontScale,
            color=(0, 0, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    def video_start(self, source):
        try:
            cap = cv2.VideoCapture(source)
            assert cap.isOpened(), "Error reading video file"

            while cap.isOpened():
                success, self.frame = cap.read()
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")
                    break
                
                self.counting_person()

                cv2.imshow("PersonCounter", self.frame)

                # q 키 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted")

        cap.release()
        cv2.destroyAllWindows()