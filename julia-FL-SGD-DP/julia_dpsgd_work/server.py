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

logging.getLogger("flwr").setLevel(logging.INFO)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LAST_FIT_METRICS = {}


def global_metrics_path() -> Path:
    return Path(getattr(parameters_federated, "GLOBAL_METRICS_PATH", "artifacts/global_metrics.csv"))


def experiment_config() -> dict:
    return {
        "run_id": RUN_ID,
        "foundation_model": parameters_federated.FOUNDATION_MODEL,
        "foundation_image_size": parameters_federated.FOUNDATION_IMAGE_SIZE,
        "embedding_dim": parameters_federated.EMBEDDING_DIM,
        "embedding_hidden_size": parameters_federated.EMBEDDING_HIDDEN_SIZE,
        "embedding_normalization": "per_client_standardization",
        "num_partitions": parameters_federated.NUM_PARTITIONS,
        "num_server_rounds": parameters_federated.NUM_SERVER_ROUNDS,
        "fraction_fit": parameters_federated.FRACTION_FIT,
        "fraction_evaluate": parameters_federated.FRACTION_EVALUATE,
        "partitioner": parameters_federated.PARTITIONER,
        "dirichlet_alpha": parameters_federated.DIRICHLET_ALPHA,
        "batch_size": parameters_federated.BATCH_SIZE,
        "classifier_epochs": parameters_federated.EPOCHS,
        "classifier_lr": parameters_federated.LR,
        "num_classes": parameters_federated.NUM_CLASSES,
        "use_local_dp_cvae": parameters_federated.USE_LOCAL_DP_CVAE,
        "target_delta": parameters_federated.TARGET_DELTA,
        "target_epsilon": parameters_federated.TARGET_EPSILON,
        "max_grad_norm": parameters_federated.MAX_GRAD_NORM,
        "cvae_hidden_dim": parameters_federated.CVAE_HIDDEN_DIM,
        "cvae_latent_dim": parameters_federated.CVAE_LATENT_DIM,
        "cvae_batch_size": parameters_federated.CVAE_BATCH_SIZE,
        "cvae_epochs": parameters_federated.CVAE_EPOCHS,
        "cvae_lr": parameters_federated.CVAE_LR,
        "cvae_beta": parameters_federated.CVAE_BETA,
        "synthetic_cache_version": parameters_federated.SYNTHETIC_CACHE_VERSION,
        "force_regenerate_synthetics": parameters_federated.FORCE_REGENERATE_SYNTHETICS,
        "retrain_cvae_every_round": parameters_federated.RETRAIN_CVAE_EVERY_ROUND,
    }


def global_metrics_fieldnames() -> list:
    return [
        "run_id",
        "round",
        "foundation_model",
        "foundation_image_size",
        "embedding_dim",
        "embedding_hidden_size",
        "embedding_normalization",
        "num_partitions",
        "num_server_rounds",
        "fraction_fit",
        "fraction_evaluate",
        "partitioner",
        "dirichlet_alpha",
        "batch_size",
        "classifier_epochs",
        "classifier_lr",
        "num_classes",
        "use_local_dp_cvae",
        "target_delta",
        "target_epsilon",
        "max_grad_norm",
        "cvae_hidden_dim",
        "cvae_latent_dim",
        "cvae_batch_size",
        "cvae_epochs",
        "cvae_lr",
        "cvae_beta",
        "synthetic_cache_version",
        "force_regenerate_synthetics",
        "retrain_cvae_every_round",
        "epsilon_cvae_mean",
        "cvae_loss_mean",
        "global_loss",
        "global_accuracy",
        "eval_noise_std_1",
        "global_loss_noisy_1",
        "global_accuracy_noisy_1",
        "accuracy_drop_noise_1",
        "eval_noise_std_2",
        "global_loss_noisy_2",
        "global_accuracy_noisy_2",
        "accuracy_drop_noise_2",
        "eval_noise_std_3",
        "global_loss_noisy_3",
        "global_accuracy_noisy_3",
        "accuracy_drop_noise_3",
    ]


def save_global_metrics(server_round, loss, accuracy, extra_metrics=None):
    path = global_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    row = experiment_config()
    row.update(LAST_FIT_METRICS)
    row.update(
        {
            "round": server_round,
            "global_loss": loss,
            "global_accuracy": accuracy,
        }
    )
    if extra_metrics:
        row.update(extra_metrics)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_metrics_fieldnames(), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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

    test_data = dataset["test"].with_transform(apply_transforms)
    return DataLoader(test_data, batch_size=parameters_federated.BATCH_SIZE)


def server_embedding_metadata(testloader, image_noise_std):
    return {
        "cache_version": 1,
        "dataset": "ylecun/mnist",
        "num_test_examples": len(testloader.dataset),
        "foundation_model": parameters_federated.FOUNDATION_MODEL,
        "foundation_image_size": parameters_federated.FOUNDATION_IMAGE_SIZE,
        "embedding_dim": parameters_federated.EMBEDDING_DIM,
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
        model = train.EmbeddingClassifier(
            input_size=parameters_federated.EMBEDDING_DIM,
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

            extra_metrics = {
                "eval_noise_std_1": parameters_federated.EVAL_GAUSSIAN_NOISE_STD1,
                "eval_noise_std_2": parameters_federated.EVAL_GAUSSIAN_NOISE_STD2,
                "eval_noise_std_3": parameters_federated.EVAL_GAUSSIAN_NOISE_STD3,
                "global_accuracy_noisy_1": accuracy_noisy1,
                "global_accuracy_noisy_2": accuracy_noisy2,
                "global_accuracy_noisy_3": accuracy_noisy3,
                "accuracy_drop_noise_1": accuracy_clean - accuracy_noisy1,
                "accuracy_drop_noise_2": accuracy_clean - accuracy_noisy2,
                "accuracy_drop_noise_3": accuracy_clean - accuracy_noisy3,
                "global_loss_noisy_1": loss_noisy1,
                "global_loss_noisy_2": loss_noisy2,
                "global_loss_noisy_3": loss_noisy3,
            }

            print(
                f"[Servidor] Avaliação final | "
                f"limpo: acc={accuracy_clean * 100:.2f}% | "
                f"ruído σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD1}: acc={accuracy_noisy1 * 100:.2f}% | "
                f"ruído σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD2}: acc={accuracy_noisy2 * 100:.2f}% | "
                f"ruído σ={parameters_federated.EVAL_GAUSSIAN_NOISE_STD3}: acc={accuracy_noisy3 * 100:.2f}%"
            )
            print(f"{accuracy_clean}  {accuracy_noisy1}  {accuracy_noisy2}  {accuracy_noisy3}")
            save_global_metrics(server_round, loss_clean, accuracy_clean, extra_metrics)
            return loss_clean, {
                "global_accuracy": accuracy_clean,
                **extra_metrics,
            }

        print(
            f"[Servidor] Avaliação global | "
            f"sem ruído: acc={accuracy_clean * 100:.2f}% | "
        )
        save_global_metrics(server_round, loss_clean, accuracy_clean)
        return loss_clean, {
            "global_accuracy": accuracy_clean,
        }

    return evaluate


def fit_metrics_aggregation_fn(metrics):
    global LAST_FIT_METRICS
    epsilons = []
    cvae_losses = []
    for _, m in metrics:
        if "epsilon_cvae" in m:
            epsilons.append(m["epsilon_cvae"])
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
    LAST_FIT_METRICS = aggregated.copy()
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
    testloader = get_test_loader("ylecun/mnist")

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
