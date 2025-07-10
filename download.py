from pathlib import Path
from ultralytics import YOLO

import requests

# Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry


def download():
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    DET_MODEL_NAME = "yolov8n"

    det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")
    label_map = det_model.model.names

    # Need to make en empty call to initialize the model
    res = det_model()
    det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
    if not det_model_path.exists():
        det_model.export(format="openvino", dynamic=True, half=True)
    
    return det_model, det_model_path