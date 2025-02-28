# coding=utf-8
# Implements API for Qwen1.5-1.8b in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.

import time
import torch
import json
import uvicorn
# uvicorn asyncio 库 异步web服务器 http websocket 协议

from pydantic import BaseModel, Field
# from pydantic import BaseModel, Field
# class User(BaseModel):
#     name: str
#     age: int = Field(default=18, description="User's age")
from fastapi import FastAPI, HTTPException
# FastAPI 
# from fastapi import FastAPI, HTTPException
# app = FastAPI()
# @app.get("/items/{item_id}")
# async def read_item(item_id: int):
#    if item_id == 3:
#        raise HTTPException(status_code=418, detail="I'm a teapot")
#    return {"item_id": item_id}
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
# 异步上下文
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from threading import Thread

# EventSourceResponse 流式数据
@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

# pydantic 的数据验证作用 确保数据安全以及符合我们的对其的要求
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]

class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    completion_tokens: Optional[int] = None

# response_model=ModelList: 这是一个参数，指定了响应模型。ModelList 应该是一个 Pydantic 模型，
# 用于定义响应数据的结构和类型。
# FastAPI 将自动将函数返回的数据序列化为 JSON，并确保它符合 ModelList 模型的结构。
# 这有助于文档生成和响应验证
def _chat_stream(input_ids, model, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    attention_mask = torch.where(input_ids == pad_token_id, 0, 1)
    attention_mask = attention_mask.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        max_length=3000,
        attention_mask=attention_mask,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    conversation = request.messages[:]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    pad_token_id = tokenizer.pad_token_id
    attention_mask = torch.where(input_ids == pad_token_id, 0, 1)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    # 流式返回答案
    if request.stream:
        generate = predict(input_ids, request.model) # input + model_id
        return EventSourceResponse(generate, media_type="text/event-stream")

    generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10000)
    response = tokenizer.decode(generated_ids[0])

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop")

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", completion_tokens=len(generated_ids[0]))


async def predict(input_ids: List[List[int]], model_id:str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    json_str = json.dumps(chunk.dict(), ensure_ascii=False)
    yield json_str
    
    current_length = 0
    
    for new_response in _chat_stream(input_ids, model, tokenizer):
        if len(new_response) == current_length:
            continue

        new_text = new_response
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        json_str = json.dumps(chunk.dict(), ensure_ascii=False)
        yield json_str


    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    json_str = json.dumps(chunk.dict(), ensure_ascii=False)
    yield json_str
    yield '[DONE]'
    
if __name__ == "__main__":
    model_path = "/root/.cache/modelscope/hub/Qwen/Qwen1.5-1.8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
    
