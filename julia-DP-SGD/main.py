"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""
from flwr.simulation import run_simulation
from server import ServerApp, server_fn
from client import ClientApp, client_fn
import parameters_federated


TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 2.0

client_resources = {
    "num_cpus": 2, 
    "num_gpus": 0.0, # Se estiver usando CPU
}

server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)

import ray
# ray.init(local_mode=True)

hist = run_simulation(
    server_app=server_app, 
    client_app=client_app, 
    num_supernodes=parameters_federated.NUM_PARTITIONS,
    backend_config={"client_resources": client_resources}
)
