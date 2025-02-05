import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings  # класс для работы с ветроной базой

from faiss_db.db import FaissVectorDataBase
from gpt_engine.gpt import generate_answer

alfa_insurance_faiss_db = FaissVectorDataBase(db_name="alfa_insurance")
embeddings = OpenAIEmbeddings()

alfa_insurance_faiss_db.db_vector = alfa_insurance_faiss_db.load_db_vector(
    embeddings_model=embeddings,
    db_folder_path="saved_db",
    index_name="alfa_insurance_faiss_db"
)
db_index = alfa_insurance_faiss_db.db_vector


class Question(BaseModel):
    text: str


def create_fastapi_app():
    REQUEST_COUNTER = 0

    app = FastAPI(title="FastAPI")

    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=["*"],
    # )

    # поддержка cors
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["Индекс"])
    async def index():
        return {'message': "API GPT"}

    @app.post("/ask", tags=["Вопрос"])
    async def ask_gpt(question: Question):
        nonlocal REQUEST_COUNTER
        REQUEST_COUNTER += 1
        # answer = db_index.similarity_search(question.text, k=5)

        answer = await generate_answer(query=question.text,  # запрос пользователя
                                       db_index=db_index,  # векторная база знаний
                                       )
        return {'message': answer}

    @app.get("/counter", tags=["Счетчик запросов"])
    async def get_counter():
        return {'request counter': REQUEST_COUNTER}

    return app


app = create_fastapi_app()

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        reload=True,
    )
