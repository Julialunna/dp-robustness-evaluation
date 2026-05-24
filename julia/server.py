"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters, NDArrays, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import parameters_to_ndarrays

import numpy as np
from pathlib import Path
import torch
import train
import parameters_federated
from train import MLP, set_weights, test
from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader

# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_accuracy = sum(accuracies) / sum(examples)
    print(f"\n---> [Servidor] Acurácia da Rodada: {aggregated_accuracy * 100:.2f}%\n")
    return {"accuracy": aggregated_accuracy}



def get_test_loader(dataset_str:str):
    dataset = load_dataset(dataset_str)
    
    pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    test_data = dataset["test"].with_transform(apply_transforms)
    testloader = DataLoader(test_data, batch_size=parameters_federated.BATCH_SIZE)
    return testloader


def get_evaluate_fn(testloader):
    def evaluate(server_round: int, parameters: NDArrays, config: dict):
        model=MLP(num_classes=10)
        set_weights(model, parameters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        loss, accuracy = test(model, testloader, device)
        return loss, {"global_accuracy": accuracy}
    return evaluate
    
def fit_metrics_aggregation_fn(metrics):
    epsilons = []

    for num_examples, m in metrics:
        if "epsilon-dp" in m:
            epsilons.append(m["epsilon-dp"])

    if len(epsilons) == 0:
        return {}

    epsilon_mean = sum(epsilons) / len(epsilons)
    print(
        f"\n---> [Servidor] Epsilon da rodada: "
        f"médio={epsilon_mean:.2f}, "

    )
    return {
        "epsilon_mean": epsilon_mean,
    }

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = parameters_federated.NUM_SERVER_ROUNDS

    ndarrays = train.get_weights(train.MLP(num_classes=parameters_federated.NUM_CLASSES))
    parameters = ndarrays_to_parameters(ndarrays)
    testloader = get_test_loader("ylecun/mnist")
    strategy = FedAvg(
        fraction_fit=parameters_federated.FRACTION_FIT,
        fraction_evaluate=parameters_federated.FRACTION_EVALUATE,
        evaluate_fn=get_evaluate_fn(testloader=testloader),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config, strategy=strategy)


# def get_evaluate_fn():
#     def evaluate(server_round: int, parameters: NDArrays, config: dict):
#         if server_round == parameters_federated.NUM_SERVER_ROUNDS:
#             print(f"\n[Servidor] Salvando modelo da última rodada ({server_round})...")
            
#             save_path = Path("./modelos/modelo_final_FL_DP.npy")
#             save_path.parent.mkdir(parents=True, exist_ok=True)
            
#             params_array = np.empty(len(parameters), dtype=object)
#             for i, arr in enumerate(parameters):
#                 params_array[i] = arr
            
#             np.save(save_path, params_array, allow_pickle=True)
#             print(f"[Servidor] Pesos salvos com sucesso em '{save_path}'")
        
#         return None
#     return evaluate
server_app = ServerApp(server_fn=server_fn)