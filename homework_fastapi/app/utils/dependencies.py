from fastapi import Header, HTTPException
from app.utils.jwt_handler import decode_token

def get_current_user(Authorization: str = Header(None)):
    if Authorization is None:
        raise HTTPException(401, "로그인이 필요합니다")

    token = Authorization.split(" ")[1]
    payload = decode_token(token)

    if not payload:
        raise HTTPException(401, "토큰이 유효하지 않습니다")

    return payload
