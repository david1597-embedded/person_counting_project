from pathlib import Path
from ultralytics import YOLO
from os import PathLike

import requests

def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    show_progress: bool = True,
) -> PathLike:
    from tqdm.notebook import tqdm_notebook
    import requests
    import urllib.parse

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    filepath = Path(directory) / filename if directory is not None else filename
    if filepath.exists():
        return filepath.resolve()

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url=url, headers={"User-agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError
    ) as error:  # For error associated with not-200 codes. Will output something like: "404 Client Error: Not Found for url: {url}"
        raise Exception(error) from None
    except requests.exceptions.Timeout:
        raise Exception(
            "Connection timed out. If you access the internet through a proxy server, please "
            "make sure the proxy is set in the shell from where you launched Jupyter."
        ) from None
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}") from None

    # download the file if it does not exist
    filesize = int(response.headers.get("Content-length", 0))
    if not filepath.exists():
        with tqdm_notebook(
            total=filesize,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=str(filename),
            disable=not show_progress,
        ) as progress_bar:
            with open(filepath, "wb") as file_object:
                for chunk in response.iter_content(chunk_size):
                    file_object.write(chunk)
                    progress_bar.update(len(chunk))
                    progress_bar.refresh()
    else:
        print(f"'{filepath}' already exists.")

    response.close()

    return filepath.resolve()

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

if __name__=="__main__":
    download()