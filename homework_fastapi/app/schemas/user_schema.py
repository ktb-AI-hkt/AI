from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    password: str
    name: str
    email: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    name: str | None = None
    email: str | None = None

class UserResponse(BaseModel):
    id: int
    username: str
    name: str
    email: str

    class Config:
        orm_mode = True
