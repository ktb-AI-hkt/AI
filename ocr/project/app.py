# app.py -> 이거 그냥 GPT가 만든거에엽
from fastapi import FastAPI, UploadFile, File, HTTPException
from pipeline import run_pipeline
import tempfile
import os

app = FastAPI(
    title="OCR → Schedule Parser API",
    description="이미지에서 일정 정보를 자동 추출하는 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 모든 출처 허용
    allow_credentials=True,   # 쿠키 포함 허용
    allow_methods=["*"],      # GET, POST, PUT, DELETE 등 모든 메서드 허용
    allow_headers=["*"],      # 모든 헤더 허용
)

@app.post("/ai/ocr")
async def ocr_api(file: UploadFile = File(...)):
    # 1️⃣ 이미지 파일 체크
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    # 2️⃣ 업로드 이미지를 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 3️⃣ 핵심: 파이프라인 호출
        result = run_pipeline(tmp_path)

        return result

    finally:
        # 4️⃣ 임시 파일 정리
        os.remove(tmp_path)
