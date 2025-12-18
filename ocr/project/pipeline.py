# from vision_ocr import vision_ocr
# from llm_parser import parse_with_llm

# def run_pipeline(image_path: str):
#     print("📌 OCR 수행 중...")
#     text = vision_ocr(image_path)

#     # print("📌 OCR 결과:")
#     # print(text)

#     print("\n📌 LLM으로 구조화 중...")
#     structured = parse_with_llm(text)

#     return structured


# if __name__ == "__main__":
#     result = run_pipeline("images/KakaoTalk_Photo_2025-12-18-09-38-41 005.jpeg")
#     print("\n✅ 최종 결과:")
#     print(result)

# pipeline.py
from vision_ocr import vision_ocr
from llm_parser import parse_with_llm

def run_pipeline(image_path: str) -> dict:
    """
    이미지 경로를 받아
    OCR → LLM 구조화를 수행
    """
    ocr_text = vision_ocr(image_path)
    result = parse_with_llm(ocr_text)
    return result
