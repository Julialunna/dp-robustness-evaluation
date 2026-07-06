"""Run Flower simulation for the DP-CVAE-local + FL-classifier pipeline.

Before Flower starts, this script optionally pretrains the frozen MLP embedding
extractor on the public/auxiliary 10% MNIST split. The remaining 90% is used by
clients only.
"""

from flwr.simulation import run_simulation
from server import ServerApp, server_fn
from client import ClientApp, client_fn

import parameters_federated
import train


client_resources = {
    "num_cpus": 2,
    "num_gpus": 0.0,  # Troque para >0 se quiser alocar GPU por cliente.
}

# Important: create/load the public-pretrained extractor before clients/server
# try to build their frozen embedding model.

server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)

hist = run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=parameters_federated.NUM_PARTITIONS,
    backend_config={"client_resources": client_resources},
)
