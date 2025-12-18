import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
너는 OCR로 추출된 한국어 공고문, 안내문, 공문을 분석하여
사용자에게 필요한 '일정 정보'와 '행동 지침'을
구조화된 JSON으로 변환하는 AI다.

출력은 반드시 JSON만 포함해야 하며,
설명, 주석, 추가 문장은 절대 포함하지 마라.

아래 JSON 스키마를 정확히 따른다.

JSON 스키마:
{
  "title": string,
  "summary": string,
  "action": string,
  "dateType": "SINGLE" | "RANGE" | "MULTIPLE",
  "startDate": "YYYY-MM-DD" | "",
  "endDate": "YYYY-MM-DD" | "",
  "dates": ["YYYY-MM-DD"]
}

필드 작성 규칙:

1. title
- 공고문의 성격이 한눈에 드러나는 제목
- 원문 제목이 있으면 최대한 보존
- 없으면 내용을 요약하여 생성

2. summary
- 이 공고로 인해 '무슨 일이 발생하는지'를 중심으로 작성
- 언제, 무엇이, 어떤 영향이 있는지 포함
- 사용자가 summary만 읽어도 상황을 이해할 수 있어야 함
- 1~2문장 이내의 자연스러운 한국어

3. action
- 이 공고를 본 사용자가 '미래에 해야 할 구체적인 행동'
- 명령형 또는 권고형 문장으로 작성
- 예:
  - "해당 시간 동안 엘리베이터 이용을 피하세요."
  - "점검 시간 이전에 필요한 이동을 완료하세요."
  - "추가 공지를 확인하세요."
- 만약 사용자가 특별히 할 행동이 없다면
  "별도의 조치는 필요하지 않습니다." 라고 작성

4. 날짜 처리 규칙
- 날짜가 하나면 dateType = "SINGLE"
- 기간이면 dateType = "RANGE"
- 여러 날짜면 dateType = "MULTIPLE"
- 모든 날짜는 ISO 형식 (YYYY-MM-DD)

5. startDate / endDate
- dateType이 "RANGE"일 때만 사용
- 그 외에는 빈 문자열 ""

6. dates
- dateType이 "SINGLE" 또는 "MULTIPLE"일 때 사용
- 없으면 빈 배열 []

중요:
- 절대 추측하지 말 것
- 문서에 명시된 정보만 사용
- 날짜가 불명확하면 unknown 처리
"""

def parse_with_llm(ocr_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ocr_text}
        ],
        temperature=0
    )

    content = response.choices[0].message.content
    return json.loads(content)
