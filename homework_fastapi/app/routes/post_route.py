from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.utils.dependencies import get_current_user
from app.schemas.post_schema import PostCreate, PostUpdate
from app.crud.crud import *

router = APIRouter(prefix="/posts")

@router.get("/")
def list_all(db: Session = Depends(get_db)):
    return list_posts(db)

@router.get("/{pid}")
def detail(pid: int, db: Session = Depends(get_db)):
    return get_post_detail(db, pid)

@router.post("/")
def create(body: PostCreate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    return create_post(db, user["id"], body)

@router.put("/{pid}")
def update(pid: int, body: PostUpdate, user=Depends(get_current_user), db: Session = Depends(get_db)):
    return update_post(db, pid, user["id"], body)
