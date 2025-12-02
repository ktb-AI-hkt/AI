from fastapi import FastAPI
from app.database.database import init_db
from app.routes.auth_route import router as auth_router
from app.routes.user_route import router as user_router
from app.routes.post_route import router as post_router

app = FastAPI()

init_db()

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(post_router)