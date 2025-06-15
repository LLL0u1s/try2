#!/bin/bash

echo "准备数据..."
python prepare_data.py

echo "启动 Flower 服务器..."
python pt_server.py &
SERVER_PID=$!

sleep 3  # 等待服务器启动

echo "启动客户端 Alice..."
python pt_client.py --split alice &
CLIENT_ALICE_PID=$!

echo "启动客户端 Bob..."
python pt_client.py --split bob &
CLIENT_BOB_PID=$!

wait $SERVER_PID $CLIENT_ALICE_PID $CLIENT_BOB_PID
