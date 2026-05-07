"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters, NDArrays, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import parameters_to_ndarrays

import numpy as np
from pathlib import Path

import train
import parameters_federated


# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_accuracy = sum(accuracies) / sum(examples)
    print(f"\n---> [Servidor] Acurácia da Rodada: {aggregated_accuracy * 100:.2f}%\n")
    return {"accuracy": aggregated_accuracy}


def get_evaluate_fn():
    def evaluate(server_round: int, parameters: NDArrays, config: dict):
        if server_round == parameters_federated.NUM_SERVER_ROUNDS:
            print(f"\n[Servidor] Salvando modelo da última rodada ({server_round})...")
            
            save_path = Path("./modelos/modelo_final_FL_DP.npy")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            params_array = np.empty(len(parameters), dtype=object)
            for i, arr in enumerate(parameters):
                params_array[i] = arr
            
            np.save(save_path, params_array, allow_pickle=True)
            print(f"[Servidor] Pesos salvos com sucesso em '{save_path}'")
        
        return None
    return evaluate

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = parameters_federated.NUM_SERVER_ROUNDS

    ndarrays = train.get_weights(train.MLP(parameters_federated.NUM_CLASSES))
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config, strategy=strategy)


server_app = ServerApp(server_fn=server_fn)