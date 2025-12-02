from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.schemas.user_schema import UserCreate, UserLogin
from app.crud.crud import create_user, login_user

router = APIRouter(prefix="/auth")

@router.post("/register")
def register(data: UserCreate, db: Session = Depends(get_db)):
    return create_user(db, data)

@router.post("/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    return login_user(db, data.username, data.password)
