from fastapi import FastAPI
from pydantic import BaseModel
from app import graph  # 从你的 app.py 中导入已编译的 graph
import uvicorn

app = FastAPI(title="企业内部智能助理 API", description="多Agent协作接口")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QueryRequest):
    """接收用户问题，返回智能助理的回答"""
    initial_state = {"question": request.question, "intent": "", "answer": ""}
    final_state = graph.invoke(initial_state)
    return {"answer": final_state["answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)