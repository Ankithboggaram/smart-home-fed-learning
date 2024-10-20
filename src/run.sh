#!/bin/bash

python run_server.py &
python run_client.py --config ./client_1.yaml &
python run_client.py --config ./client_2.yaml &
python run_client.py --config ./client_3.yaml &
python run_client.py --config ./client_4.yaml &
python run_client.py --config ./client_5.yaml
wait