from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

import parameters_federated
from client import client_fn
from server import server_fn


def reset_global_metrics():
    metrics_path = Path(getattr(parameters_federated, "GLOBAL_METRICS_PATH", "artifacts/global_metrics.csv"))
    if metrics_path.exists():
        metrics_path.unlink()


def save_global_metric_plots():
    metrics_path = Path(getattr(parameters_federated, "GLOBAL_METRICS_PATH", "artifacts/global_metrics.csv"))
    if not metrics_path.exists():
        return

    df = pd.read_csv(metrics_path)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    if "round" in df.columns and "global_loss" in df.columns:
        plt.plot(df["round"], df["global_loss"], marker="o")
        plt.xlabel("Rodada")
        plt.ylabel("Loss global")
        plt.title("Loss global por rodada")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("artifacts/global_loss.png", dpi=300)
        plt.close()

    if "round" in df.columns and "global_accuracy" in df.columns:
        plt.plot(df["round"], df["global_accuracy"], marker="o")
        plt.xlabel("Rodada")
        plt.ylabel("Acurácia global")
        plt.title("Acurácia global por rodada")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("artifacts/global_accuracy.png", dpi=300)
        plt.close()


reset_global_metrics()

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

save_global_metric_plots()
