from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.data_model import User, Post
from app.schemas.user_schema import UserCreate, UserUpdate
from app.schemas.post_schema import PostCreate, PostUpdate
from app.utils.password_handler import hash_password, verify_password
from app.utils.jwt_handler import create_token


# ----------------------------------------
# USER CRUD
# ----------------------------------------

def create_user(db: Session, data: UserCreate):
    exists = db.query(User).filter(User.username == data.username).first()
    if exists:
        raise HTTPException(400, "이미 존재하는 사용자명입니다.")

    user = User(
        username=data.username,
        password=hash_password(data.password),
        name=data.name,
        email=data.email
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(400, "아이디 또는 비밀번호가 틀렸습니다.")

    if not verify_password(password, user.password):
        raise HTTPException(400, "아이디 또는 비밀번호가 틀렸습니다.")

    token = create_token({"id": user.id})
    return {"token": token}


def get_user(db: Session, uid: int):
    return db.query(User).filter(User.id == uid).first()


def update_user(db: Session, uid: int, data: UserUpdate):
    user = get_user(db, uid)
    if not user:
        raise HTTPException(404, "User not found")

    for k, v in data.dict(exclude_none=True).items():
        setattr(user, k, v)

    db.commit()
    db.refresh(user)
    return user



# ----------------------------------------
# POST CRUD
# ----------------------------------------

def list_posts(db: Session):
    return db.query(Post).all()


def get_post_detail(db: Session, pid: int):
    post = db.query(Post).filter(Post.id == pid).first()
    if not post:
        raise HTTPException(404, "Post not found")
    return post


def create_post(db: Session, author_id: int, data: PostCreate):
    post = Post(
        title=data.title,
        content=data.content,
        author_id=author_id
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


def update_post(db: Session, pid: int, author_id: int, data: PostUpdate):
    post = get_post_detail(db, pid)

    if post.author_id != author_id:
        raise HTTPException(403, "수정 권한이 없습니다.")

    for k, v in data.dict(exclude_none=True).items():
        setattr(post, k, v)

    db.commit()
    db.refresh(post)
    return post
