"""Flower server for FL classifier trained on local DP-CVAE synthetic embeddings."""

import logging
from typing import List, Tuple

import torch
from datasets import load_dataset
from flwr.common import Context, Metrics, NDArrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

import parameters_federated
import train

logging.getLogger("flwr").setLevel(logging.INFO)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_accuracy = sum(accuracies) / max(sum(examples), 1)
    print(f"\n---> [Servidor] Acurácia agregada dos clientes: {aggregated_accuracy * 100:.2f}%\n")
    return {"accuracy": aggregated_accuracy}


def get_test_loader(dataset_str: str):
    dataset = load_dataset(dataset_str)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((parameters_federated.MEAN,), (parameters_federated.STD,))]
    )

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    test_data = dataset["test"].with_transform(apply_transforms)
    return DataLoader(test_data, batch_size=parameters_federated.BATCH_SIZE)




def get_evaluate_fn(testloader):
    clean_eval_loader = None
    noisy_eval_loader = None

    def evaluate(server_round: int, parameters: NDArrays, config: dict):
        nonlocal clean_eval_loader, noisy_eval_loader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = train.EmbeddingClassifier(
            input_size=parameters_federated.EMBEDDING_DIM,
            hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
            num_classes=parameters_federated.NUM_CLASSES,
        )
        train.set_weights(model, parameters)
    
        if clean_eval_loader is None or noisy_eval_loader is None:
            embedding_model = train.build_embedding_extractor(device)

            clean_embeddings, clean_labels = train.extract_embeddings_from_loader(
                embedding_model,
                testloader,
                device,
                image_noise_std=0.0,
            )

            noisy_embeddings, noisy_labels = train.extract_embeddings_from_loader(
                embedding_model,
                testloader,
                device,
                image_noise_std=parameters_federated.EVAL_GAUSSIAN_NOISE_STD,
                image_noise_seed=parameters_federated.EVAL_GAUSSIAN_NOISE_SEED,
            )

            clean_eval_loader = train.make_embedding_loader(
                clean_embeddings,
                clean_labels,
                batch_size=parameters_federated.BATCH_SIZE,
                shuffle=False,
            )

            noisy_eval_loader = train.make_embedding_loader(
                noisy_embeddings,
                noisy_labels,
                batch_size=parameters_federated.BATCH_SIZE,
                shuffle=False,
            )


        loss_clean, accuracy_clean = train.test_embedding_classifier_on_loader(
            model,
            clean_eval_loader,
            device,
        )

        loss_noisy, accuracy_noisy = train.test_embedding_classifier_on_loader(
            model,
            noisy_eval_loader,
            device,
        )
        print(
            f"[Servidor] Avaliação global | "
            f"sem ruído: acc={accuracy_clean * 100:.2f}% | "
            f"com ruído: acc={accuracy_noisy * 100:.2f}%"
        )

        return loss_clean, {
            "global_accuracy": accuracy_clean,
            "global_accuracy_noisy": accuracy_noisy,
            "accuracy_drop_noise": accuracy_clean - accuracy_noisy,
            "global_loss_noisy": loss_noisy,
            "eval_noise_std": parameters_federated.EVAL_GAUSSIAN_NOISE_STD,
        }

    return evaluate



def fit_metrics_aggregation_fn(metrics):
    epsilons = []
    cvae_losses = []
    for _, m in metrics:
        if "epsilon-dp-cvae" in m:
            epsilons.append(m["epsilon-dp-cvae"])
        if "cvae_loss" in m:
            cvae_losses.append(m["cvae_loss"])

    aggregated = {}
    if epsilons:
        epsilon_mean = sum(epsilons) / len(epsilons)
        aggregated["epsilon_cvae_mean"] = epsilon_mean
        print(f"[Servidor] ε médio dos CVAEs locais: {epsilon_mean:.2f}")
    if cvae_losses:
        cvae_loss_mean = sum(cvae_losses) / len(cvae_losses)
        aggregated["cvae_loss_mean"] = cvae_loss_mean
        print(f"[Servidor] Loss média dos CVAEs locais: {cvae_loss_mean:.4f}")
    return aggregated


def server_fn(context: Context) -> ServerAppComponents:
    ndarrays = train.get_weights(
        train.EmbeddingClassifier(
            input_size=parameters_federated.EMBEDDING_DIM,
            hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
            num_classes=parameters_federated.NUM_CLASSES,
        )
    )
    parameters = ndarrays_to_parameters(ndarrays)
    testloader = train.get_test_loader("ylecun/mnist")

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
