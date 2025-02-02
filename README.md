# PowerNet: Federated Learning for Energy Consumption Forecasting

Our project aims to forecast electricity consumption in a neighborhood using smart home data from multiple houses through a federated learning framework. This approach enables model training on distributed data while preserving privacy. Additionally, we utilize Bayesian optimization to determine the ideal hyperparameters for the neural network, enhancing the model's performance.

The results are discussed in the [PowerNet](PowerNet.pdf) document.

This document specifies instructions on how to set the system up for federated learning experiments.

## Setup Details

Our experiments were conducted on a multi-client architecture where each client simulates a smart home. Each client is responsible for training its own model on local datasets. 

To set up the project, please create a virual environment and install required dependencies by running:
```bash
pip install -r requirements.txt
```
This will download the necessary dependencies and set up the environment for federated learning.

On your main node, execute the following command:

```bash
python run_server.py
```

This will start the server for federated learning. On each client node, run:
```
python run_client.py --config ./client_[CLIENT_ID].yaml 
```
Make sure to replace [CLIENT_ID] with the corresponding ID for each client. Our project involves 4 clients

Once the commands have been successfully executed on all the nodes, the federated learning system will be up and running. By default, each client will train a CNN model on its local dataset.

The entire setup can be simulated on a single machine by running the following commands:
```bash
chmod +x run.sh
./run.sh
```
Note that all the files for federated learning are present in the [src](/src) directory.

To adjust the model or the training configuration, modify the settings in config.py.

## Future Work
There is still significant room for improvement in the project. The current implementation could benefit from enhanced configuration management. We plan to streamline the configuration process, allowing paths and other settings to be dynamically loaded from environment variables.

Additionally, integrating more sophisticated models or exploring different federated learning strategies are potential areas for future research. We welcome contributions and suggestions to improve the project!
