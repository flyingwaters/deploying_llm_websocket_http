import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

print("基于Qwen 1.5 1.8B的人工智能聊天助手（PS: 输入 exit 退出对话。）")
while True:
    user_input = input("User: \n")
    if user_input.lower() == 'exit':
        print("退出程序。")
        break
    print("AI Assistant:")
    for chunk in openai.ChatCompletion.create(
        model="qwen_1.5_1.8b_chat",
        messages=[
            {"role": "user", "content": user_input}
        ],
        max_length=5000,
            stream=True ):
        
        if hasattr(chunk.choices[0].delta, "content"):
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
    print()