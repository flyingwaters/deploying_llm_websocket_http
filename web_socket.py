from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import time
import torch
import uvicorn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = "/root/.cache/modelscope/hub/Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True).half().cuda()
model = model.eval()

def _chat_stream(query, history):
    conversation = history
    all_his = [i["content"] for i in conversation]
    his_len = "".join(all_his)
    if len(his_len)>5000:
        conversation = []
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    input_ids = inputs.to(model.device)
    pad_token_id = tokenizer.pad_token_id
    attention_mask = torch.where(input_ids == pad_token_id, 0, 1)
    attention_mask = attention_mask.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        max_length = 10000,
        attention_mask=attention_mask,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware
)

with open('websocket_demo.html') as f:
    html = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
    await websocket.accept()
    try:
        while True:
            json_request = await websocket.receive_json()
            query = json_request['query']
            history = json_request['history']
            final_response = ""
            for response in _chat_stream(query=query, history=history):
                final_response+=response
                history.append({"role":"assistant","content":final_response})
                await websocket.send_json({
                    "response": response,
                    "history": history,
                    "status": 202,
                })
            print(final_response)
            await websocket.send_json({"status": 200})
    except WebSocketDisconnect:
        pass


def main():
    uvicorn.run(f"{__name__}:app", host='0.0.0.0', port=8000, workers=1)


if __name__ == '__main__':
    main()