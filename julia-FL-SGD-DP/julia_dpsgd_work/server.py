import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset
from flwr.common import Context, Metrics, NDArrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

import parameters_federated
import train
import models

logging.getLogger("flwr").setLevel(logging.INFO)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LAST_FIT_METRICS = {}

def get_dataset_name():
    if parameters_federated.DATASET == "mnist":
        return "ylecun/mnist"

    return f"danjacobellis/{parameters_federated.DATASET}_224"


#media ponderada das acuracias dos clientes, ponderada pelo número de exemplos de cada cliente, assim como a loss 
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_accuracy = sum(accuracies) / max(sum(examples), 1)
    print(f"\n---> [Servidor] Acurácia agregada dos clientes: {aggregated_accuracy * 100:.2f}%\n")
    return {"accuracy": aggregated_accuracy}


def get_test_loader(dataset_str: str):
    dataset = load_dataset(dataset_str)
    pytorch_transforms = Compose([ToTensor()])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    test_data = train.preprocess_dataset(dataset["test"])
    test_data = test_data.with_transform(apply_transforms)
    return DataLoader(test_data, batch_size=parameters_federated.BATCH_SIZE)


def server_embedding_metadata(testloader, image_noise_std):
    embedding_dim = (
        384
        if parameters_federated.FOUNDATION_MODEL == "dinov2_s"
        else parameters_federated.EMBEDDING_DIM
    )

    return {
        "cache_version": 1,
        "dataset": get_dataset_name(),
        "num_test_examples": len(testloader.dataset),
        "foundation_model": parameters_federated.FOUNDATION_MODEL,
        "foundation_image_size": parameters_federated.FOUNDATION_IMAGE_SIZE,
        "embedding_dim": parameters_federated.EMBEDDING_DIM,
        "embedding_dim": embedding_dim,
        "image_noise_std": image_noise_std,
        "image_noise_seed": parameters_federated.EVAL_GAUSSIAN_NOISE_SEED,
    }


def get_cached_server_embedding_loader(
    embedding_model,
    testloader,
    device,
    image_noise_std,
    cache_name,
):
    return train.build_or_load_embedding_loader(
        Path("artifacts/test_embeddings") / cache_name,
        embedding_model,
        testloader,
        device,
        server_embedding_metadata(testloader, image_noise_std),
        parameters_federated.BATCH_SIZE,
        image_noise_std=image_noise_std,
        image_noise_seed=parameters_federated.EVAL_GAUSSIAN_NOISE_SEED,
    )


def get_evaluate_fn(testloader):
    embedding_models = {}

    def evaluate(server_round: int, parameters: NDArrays, config: dict):
        device = train.get_device()
    
        model = models.EmbeddingClassifier(
            hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
            num_classes=parameters_federated.NUM_CLASSES,
        )
        train.set_weights(model, parameters)
        device_key = str(device)
        if device_key not in embedding_models:
            embedding_models[device_key], _ = train.define_foundation_extractor(
                parameters_federated.FOUNDATION_MODEL,
                device,
            )
        embedding_model = embedding_models[device_key]

        clean_eval_loader = get_cached_server_embedding_loader(
            embedding_model,
            testloader,
            device,
            0.0,
            "server_clean_test.pt",
        )

        loss_clean, accuracy_clean = train.test(
            model,
            clean_eval_loader,
            device,
        )

        if server_round == parameters_federated.NUM_SERVER_ROUNDS:
            noisy_eval_loader1 = get_cached_server_embedding_loader(
                embedding_model,
                testloader,
                device,
                parameters_federated.EVAL_GAUSSIAN_NOISE_STD1,
                "server_noise_1_test.pt",
            )
            loss_noisy1, accuracy_noisy1 = train.test(
                model,
                noisy_eval_loader1,
                device,
            )

            noisy_eval_loader2 = get_cached_server_embedding_loader(
                embedding_model,
                testloader,
                device,
                parameters_federated.EVAL_GAUSSIAN_NOISE_STD2,
                "server_noise_2_test.pt",
            )
            loss_noisy2, accuracy_noisy2 = train.test(
                model,
                noisy_eval_loader2,
                device,
            )

            noisy_eval_loader3 = get_cached_server_embedding_loader(
                embedding_model,
                testloader,
                device,
                parameters_federated.EVAL_GAUSSIAN_NOISE_STD3,
                "server_noise_3_test.pt",
            )
            loss_noisy3, accuracy_noisy3 = train.test(
                model,
                noisy_eval_loader3,
                device,
            )

            use_dp = parameters_federated.USE_LOCAL_DP_CVAE

            if use_dp:
                epsilon_description = str(parameters_federated.TARGET_EPSILON)
            else:
                epsilon_description = "sem DP"
                
            drop_sigma1 = accuracy_clean - accuracy_noisy1
            drop_sigma2 = accuracy_clean - accuracy_noisy2
            drop_sigma3 = accuracy_clean - accuracy_noisy3
            retention_sigma1 = accuracy_noisy1 / accuracy_clean
            retention_sigma2 = accuracy_noisy2 / accuracy_clean
            retention_sigma3 = accuracy_noisy3 / accuracy_clean

            print(
                f"[Servidor] Avaliação final | "
                f"dataset={parameters_federated.DATASET} | "
                f"DP-CVAE={use_dp} | "
                f"epsilon={epsilon_description} | "
                f"extrator={parameters_federated.FOUNDATION_MODEL} | "
                f"distribuição={parameters_federated.PARTITIONER} | "
                f"alpha={parameters_federated.DIRICHLET_ALPHA}\n"
                f"limpo: acc={accuracy_clean * 100:.2f}% | "
                f"ruído σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD1}: "
                f"acc={accuracy_noisy1 * 100:.2f}% | "
                f"ruído σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD2}: "
                f"acc={accuracy_noisy2 * 100:.2f}% | "
                f"ruído σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD3}: "
                f"acc={accuracy_noisy3 * 100:.2f}%"
            )
            
            print(
                f"[Servidor] Retenção de acurácia com ruído | "
                f"σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD1}: "
                f"{retention_sigma1 * 100:.2f}% | {drop_sigma1 * 100:.2f}"
                f"σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD2}: "
                f"{retention_sigma2 * 100:.2f}% | {drop_sigma2 * 100:.2f}"
                f"σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD3}: "
                f"{retention_sigma3 * 100:.2f}% | {drop_sigma3 * 100:.2f}%"
            )

            return loss_clean, {
                "global_accuracy": accuracy_clean,
                "global_accuracy_noisy_1": accuracy_noisy1,
                "global_accuracy_noisy_2": accuracy_noisy2,
                "global_accuracy_noisy_3": accuracy_noisy3,
            }

        print(
            f"[Servidor] Avaliação global | "
            f"sem ruído: acc={accuracy_clean * 100:.2f}% | "
        )
        return loss_clean, {
            "global_accuracy": accuracy_clean,
        }

    return evaluate


def fit_metrics_aggregation_fn(metrics):
    global LAST_FIT_METRICS
    cvae_losses = []
    for _, m in metrics:
        if "cvae_loss" in m:
            cvae_losses.append(m["cvae_loss"])

    aggregated = {}

    if cvae_losses:
        cvae_loss_mean = sum(cvae_losses) / len(cvae_losses)
        aggregated["cvae_loss_mean"] = cvae_loss_mean
        print(f"[Servidor] Loss média dos CVAEs locais: {cvae_loss_mean:.4f}")
    LAST_FIT_METRICS = aggregated.copy()
    return aggregated


def server_fn(context: Context) -> ServerAppComponents:
    ndarrays = train.get_weights(
        models.EmbeddingClassifier(
            hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
            num_classes=parameters_federated.NUM_CLASSES,
        )
    )
    parameters = ndarrays_to_parameters(ndarrays)
    dataset_name = get_dataset_name()
    testloader = get_test_loader(dataset_name)

    strategy = FedAvg(
        fraction_fit=parameters_federated.FRACTION_FIT,
        min_fit_clients=parameters_federated.NUM_PARTITIONS,
        min_available_clients=parameters_federated.NUM_PARTITIONS,
        fraction_evaluate=parameters_federated.FRACTION_EVALUATE,
        min_evaluate_clients=parameters_federated.NUM_PARTITIONS,
        evaluate_fn=get_evaluate_fn(testloader=testloader),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=parameters_federated.NUM_SERVER_ROUNDS)
    return ServerAppComponents(config=config, strategy=strategy)


server_app = ServerApp(server_fn=server_fn)
logging.getLogger("flwr").setLevel(logging.ERROR)
