from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.utils.dependencies import get_current_user
from app.schemas.user_schema import UserUpdate
from app.crud.crud import get_user, update_user

router = APIRouter(prefix="/user")

@router.get("/me")
def get_me(user=Depends(get_current_user), db: Session = Depends(get_db)):
    return get_user(db, user["id"])

@router.put("/me")
def update_me(body: UserUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    return update_user(db, user["id"], body)
