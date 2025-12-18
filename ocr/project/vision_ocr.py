# vision_ocr.py
import os
from dotenv import load_dotenv
from google.cloud import vision
from google.api_core.client_options import ClientOptions

load_dotenv()

def vision_ocr(image_path: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key, "GOOGLE_API_KEY not found"

    client = vision.ImageAnnotatorClient(
        client_options=ClientOptions(api_key=api_key)
    )

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise RuntimeError(response.error.message)

    return response.text_annotations[0].description if response.text_annotations else ""
