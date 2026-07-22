from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import torch
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

import parameters_federated
import train



def clear_cache_on_start():
    synthetic_cache = Path(parameters_federated.SYNTHETIC_CACHE_DIR)
    if synthetic_cache.exists():
        shutil.rmtree(synthetic_cache)
        print(f"Cache sintético apagado: {synthetic_cache}")

    test_cache = Path("artifacts/test_embeddings")
    if test_cache.exists():
        shutil.rmtree(test_cache)
        print(f"Cache de teste apagado: {test_cache}")


clear_cache_on_start()
train.ensure_embedding_extractor_ready()

from server import server_fn
from client import client_fn

client_resources = {
    "num_cpus": 2,
    "num_gpus": 1.0 if torch.cuda.is_available() else 0.0,
}

server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)

hist = run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=parameters_federated.NUM_PARTITIONS,
    backend_config={"client_resources": client_resources},
)

metrics_path = Path("artifacts/global_metrics.csv")
if metrics_path.exists():
    dataframe = pd.read_csv(metrics_path)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    plt.plot(dataframe["round"], dataframe["global_loss"], marker="o")
    plt.xlabel("Rodada")
    plt.ylabel("Loss global")
    plt.title("Loss global por rodada")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/global_loss.png", dpi=300)
    plt.close()
