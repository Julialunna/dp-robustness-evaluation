"""
Módulo do Servidor Federado (Flower Server).

Gerencia:
1. Agregação de parâmetros dos clientes via FedAvg.
2. Avaliação global a cada rodada (Acurácia, Acurácia Balanceada, F1-Score).
3. Avaliação de robustez contra ruído gaussiano (σ = 0.10, 0.25, 0.50) com taxas de retenção.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from flwr.common import Context, Metrics, NDArrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

import config
import dataset
import extractor
import models
import utils

logging.getLogger("flwr").setLevel(logging.INFO)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Média ponderada das métricas dos clientes (Acurácia, Balanced Acc, F1-Score)."""
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]
    total_examples = sum(examples)

    if total_examples == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "f1_macro": 0.0}

    aggregated = {}
    for metric_name in ["accuracy", "balanced_accuracy", "f1_macro"]:
        values = [
            num_examples * m[metric_name]
            for num_examples, m in metrics
            if metric_name in m
        ]
        if values:
            aggregated[metric_name] = sum(values) / total_examples

    acc = aggregated.get("accuracy", 0.0) * 100
    bal_acc = aggregated.get("balanced_accuracy", 0.0) * 100
    f1 = aggregated.get("f1_macro", 0.0) * 100
    print(f"\n---> [Servidor] Acurácia agregada: {acc:.2f}% | Bal Acc: {bal_acc:.2f}% | F1: {f1:.2f}%\n")
    return aggregated


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Agrega métricas reportadas no fit dos clientes (ex: loss média do DP-CVAE)."""
    cvae_losses = [m["cvae_loss"] for _, m in metrics if "cvae_loss" in m]
    aggregated = {}

    if cvae_losses:
        mean_cvae_loss = sum(cvae_losses) / len(cvae_losses)
        aggregated["cvae_loss_mean"] = mean_cvae_loss
        print(f"[Servidor] Loss média dos CVAEs locais: {mean_cvae_loss:.4f}")

    return aggregated


def _server_embedding_metadata(test_loader, image_noise_std: float) -> dict:
    return {
        "cache_version": 2,
        "dataset": dataset.get_dataset_name(),
        "num_test_examples": len(test_loader.dataset),
        "extractor_name": config.EXTRACTOR_NAME,
        "image_size": config.IMAGE_SIZE,
        "embedding_dim": config.EMBEDDING_DIM,
        "image_noise_std": image_noise_std,
        "image_noise_seed": config.EVAL_GAUSSIAN_NOISE_SEED,
    }


def _get_server_embedding_loader(
    feature_extractor,
    test_loader,
    device: torch.device,
    image_noise_std: float,
    cache_filename: str,
):
    cache_path = Path(config.TEST_EMBEDDINGS_CACHE_DIR) / cache_filename
    metadata = _server_embedding_metadata(test_loader, image_noise_std)
    return extractor.build_or_load_embedding_loader(
        cache_path=cache_path,
        extractor=feature_extractor,
        image_loader=test_loader,
        device=device,
        metadata=metadata,
        batch_size=config.BATCH_SIZE,
        image_noise_std=image_noise_std,
        image_noise_seed=config.EVAL_GAUSSIAN_NOISE_SEED,
    )


def get_evaluate_fn(test_loader):
    """Retorna o callback de avaliação centralizada do servidor."""
    cached_extractors: Dict[str, extractor.BaseFeatureExtractor] = {}

    def evaluate(server_round: int, parameters: NDArrays, config_dict: dict):
        device = utils.get_device()
        device_key = str(device)

        if device_key not in cached_extractors:
            cached_extractors[device_key] = extractor.get_feature_extractor(
                name=config.EXTRACTOR_NAME,
                device=device,
            )
        feat_extractor = cached_extractors[device_key]

        model = models.EmbeddingClassifier(
            input_dim=feat_extractor.embedding_dim,
            hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
        )
        models.set_weights(model, parameters)

        # Avaliação limpa (sem ruído: σ = 0.0)
        clean_loader = _get_server_embedding_loader(
            feat_extractor, test_loader, device, 0.0, "server_clean_test.pt"
        )
        loss_clean, metrics_clean = models.evaluate_classifier(
            model, clean_loader, device, num_classes=config.NUM_CLASSES
        )
        acc_clean = metrics_clean["accuracy"]
        bal_acc_clean = metrics_clean["balanced_accuracy"]
        f1_clean = metrics_clean["f1_macro"]

        # Avaliação de robustez a ruído apenas na rodada final
        if server_round == config.NUM_SERVER_ROUNDS:
            # Ruído Nível 1 (σ = 0.10)
            noisy_loader_1 = _get_server_embedding_loader(
                feat_extractor, test_loader, device, config.EVAL_GAUSSIAN_NOISE_STD1, "server_noise_010_test.pt"
            )
            _, metrics_noisy_1 = models.evaluate_classifier(
                model, noisy_loader_1, device, num_classes=config.NUM_CLASSES
            )
            acc_n1 = metrics_noisy_1["accuracy"]
            bal_acc_n1 = metrics_noisy_1["balanced_accuracy"]
            f1_n1 = metrics_noisy_1["f1_macro"]
            ret_acc_1 = (acc_n1 / acc_clean * 100) if acc_clean > 0 else 0.0
            ret_f1_1 = (f1_n1 / f1_clean * 100) if f1_clean > 0 else 0.0

            # Ruído Nível 2 (σ = 0.25)
            noisy_loader_2 = _get_server_embedding_loader(
                feat_extractor, test_loader, device, config.EVAL_GAUSSIAN_NOISE_STD2, "server_noise_025_test.pt"
            )
            _, metrics_noisy_2 = models.evaluate_classifier(
                model, noisy_loader_2, device, num_classes=config.NUM_CLASSES
            )
            acc_n2 = metrics_noisy_2["accuracy"]
            bal_acc_n2 = metrics_noisy_2["balanced_accuracy"]
            f1_n2 = metrics_noisy_2["f1_macro"]
            ret_acc_2 = (acc_n2 / acc_clean * 100) if acc_clean > 0 else 0.0
            ret_f1_2 = (f1_n2 / f1_clean * 100) if f1_clean > 0 else 0.0

            # Ruído Nível 3 (σ = 0.50)
            noisy_loader_3 = _get_server_embedding_loader(
                feat_extractor, test_loader, device, config.EVAL_GAUSSIAN_NOISE_STD3, "server_noise_050_test.pt"
            )
            _, metrics_noisy_3 = models.evaluate_classifier(
                model, noisy_loader_3, device, num_classes=config.NUM_CLASSES
            )
            acc_n3 = metrics_noisy_3["accuracy"]
            bal_acc_n3 = metrics_noisy_3["balanced_accuracy"]
            f1_n3 = metrics_noisy_3["f1_macro"]
            ret_acc_3 = (acc_n3 / acc_clean * 100) if acc_clean > 0 else 0.0
            ret_f1_3 = (f1_n3 / f1_clean * 100) if f1_clean > 0 else 0.0

            mode_info = f"Modo={config.TRAIN_MODE}"
            if config.USE_LOCAL_DP_CVAE:
                mode_info += f" (ε={config.TARGET_EPSILON})"

            print(
                f"\n[Servidor] Avaliação Final de Robustez | Dataset={config.DATASET} | {mode_info}\n"
                f"  Limpo (σ=0.0): Acc={acc_clean * 100:.2f}% | Bal Acc={bal_acc_clean * 100:.2f}% | F1={f1_clean * 100:.2f}%\n"
                f"  Ruído σ={config.EVAL_GAUSSIAN_NOISE_STD1}: Acc={acc_n1 * 100:.2f}% (Ret={ret_acc_1:.1f}%) | F1={f1_n1 * 100:.2f}% (Ret={ret_f1_1:.1f}%)\n"
                f"  Ruído σ={config.EVAL_GAUSSIAN_NOISE_STD2}: Acc={acc_n2 * 100:.2f}% (Ret={ret_acc_2:.1f}%) | F1={f1_n2 * 100:.2f}% (Ret={ret_f1_2:.1f}%)\n"
                f"  Ruído σ={config.EVAL_GAUSSIAN_NOISE_STD3}: Acc={acc_n3 * 100:.2f}% (Ret={ret_acc_3:.1f}%) | F1={f1_n3 * 100:.2f}% (Ret={ret_f1_3:.1f}%)"
            )

            return loss_clean, {
                "global_accuracy": acc_clean,
                "global_balanced_accuracy": bal_acc_clean,
                "global_f1_macro": f1_clean,
                # Ruído 0.10
                "accuracy_noise_010": acc_n1,
                "balanced_accuracy_noise_010": bal_acc_n1,
                "f1_noise_010": f1_n1,
                "retention_accuracy_010": ret_acc_1,
                "retention_f1_010": ret_f1_1,
                # Ruído 0.25
                "accuracy_noise_025": acc_n2,
                "balanced_accuracy_noise_025": bal_acc_n2,
                "f1_noise_025": f1_n2,
                "retention_accuracy_025": ret_acc_2,
                "retention_f1_025": ret_f1_2,
                # Ruído 0.50
                "accuracy_noise_050": acc_n3,
                "balanced_accuracy_noise_050": bal_acc_n3,
                "f1_noise_050": f1_n3,
                "retention_accuracy_050": ret_acc_3,
                "retention_f1_050": ret_f1_3,
            }

        print(
            f"[Servidor] Rodada {server_round}/{config.NUM_SERVER_ROUNDS} | "
            f"Acc: {acc_clean * 100:.2f}% | Bal Acc: {bal_acc_clean * 100:.2f}% | F1: {f1_clean * 100:.2f}%"
        )
        return loss_clean, {
            "global_accuracy": acc_clean,
            "global_balanced_accuracy": bal_acc_clean,
            "global_f1_macro": f1_clean,
        }

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    """Configura e inicializa a ServerApp do Flower com FedAvg."""
    initial_model = models.EmbeddingClassifier(
        input_dim=config.EMBEDDING_DIM,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
    )
    initial_parameters = ndarrays_to_parameters(models.get_weights(initial_model))

    test_loader = dataset.get_server_test_loader()

    strategy = FedAvg(
        fraction_fit=config.FRACTION_FIT,
        min_fit_clients=config.NUM_PARTITIONS,
        min_available_clients=config.NUM_PARTITIONS,
        fraction_evaluate=config.FRACTION_EVALUATE,
        min_evaluate_clients=config.NUM_PARTITIONS,
        evaluate_fn=get_evaluate_fn(test_loader=test_loader),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    server_config = ServerConfig(num_rounds=config.NUM_SERVER_ROUNDS)
    return ServerAppComponents(config=server_config, strategy=strategy)


server_app = ServerApp(server_fn=server_fn)
