## Deploy the qwen1.5_1.8b_chat model
#### 部署
+ 实测ubuntu 20 和 python==3.10 环境下运行有效
+ 安装依赖 pip install requirements.txt
+ llm模型下载, 在两个脚本中设置model的路径
+ python openai_test_api.py 部署 http 
+ python web_socket.py 部署 websocket

#### 客户端
+ client: client_for_api.py 用于 http 部署 验证客户端
+ websocket_demo.html: websocket_demo.html 用于验证websocket的客户端

