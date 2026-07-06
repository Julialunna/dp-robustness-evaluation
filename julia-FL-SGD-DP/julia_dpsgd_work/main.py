from flwr.simulation import run_simulation
from flwr.server import ServerApp
from flwr.client import ClientApp

from server import server_fn
from client import client_fn

import parameters_federated

import parameters_federated
import matplotlib.pyplot as plt
from pathlib import Path


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

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("artifacts/global_metrics.csv")

Path("artifacts").mkdir(parents=True, exist_ok=True)

plt.plot(df["round"], df["global_loss"], marker="o")
plt.xlabel("Rodada")
plt.ylabel("Loss global")
plt.title("Loss global por rodada")
plt.grid(True)
plt.tight_layout()
plt.savefig("artifacts/global_loss.png", dpi=300)
plt.close()