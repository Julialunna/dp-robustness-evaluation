"""
Ponto de Entrada Principal (Main Experiment Runner).

Executa uma simulação federada com controle estrito de sementes e argumentos CLI,
coletando métricas detalhadas (Acurácia, Balanced Accuracy, F1-Score e Robustez a Ruído).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

import config
import dataset
import server
import utils
from client import client_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation Runner")
    parser.add_argument("--dataset", type=str, default=config.DATASET, help="Nome do dataset")
    parser.add_argument("--partitioner", type=str, default=config.PARTITIONER, choices=["iid", "dirichlet"], help="Tipo de partição")
    parser.add_argument("--dirichlet-alpha", type=float, default=config.DIRICHLET_ALPHA, help="Alpha para partição Dirichlet")
    parser.add_argument("--mode", type=str, default=config.TRAIN_MODE, choices=["baseline", "cvae_sem_dp", "cvae_dp", "cvae_decoder_no_dp", "cvae_decoder_dp"], help="Modo de treino")
    parser.add_argument("--epsilon", type=float, default=config.TARGET_EPSILON, help="Epsilon para DP-SGD")
    parser.add_argument("--seed", type=int, default=config.BASE_SEED, help="Semente para reprodutibilidade")
    parser.add_argument("--num-rounds", type=int, default=config.NUM_SERVER_ROUNDS, help="Número de rodadas federadas")
    parser.add_argument("--output-dir", type=str, default=None, help="Diretório de saída para os resultados")
    return parser.parse_args()


def make_seeded_server_fn(base_server_fn, run_seed: int):
    """Garante que a inicialização do servidor use a semente da repetição atual."""
    def seeded_server_fn(context):
        utils.seed_all(run_seed)
        return base_server_fn(context)
    return seeded_server_fn


def make_seeded_client_fn(base_client_fn, run_seed: int):
    """Garante que cada cliente Ray seja inicializado com sua semente exclusiva."""
    worker_seeded = False

    def seeded_client_fn(context):
        nonlocal worker_seeded
        if not worker_seeded:
            partition_id = int(context.node_config["partition-id"])
            utils.seed_all(run_seed + 10_000 + partition_id)
            worker_seeded = True
        return base_client_fn(context)
    return seeded_client_fn


def clear_synthetic_cache() -> None:
    """Limpa o cache de embeddings sintéticos gerados pelos clientes."""
    cache_path = Path(config.SYNTHETIC_CACHE_DIR)
    if cache_path.exists():
        shutil.rmtree(cache_path)


def get_output_root() -> Path:
    """Diretório de saída portátil (Mac/Linux). Pode ser sobrescrito por FL_OUTPUT_ROOT."""
    output_root = Path(os.environ.get("FL_OUTPUT_ROOT", "output")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def write_run_metrics(summary: dict[str, Any]) -> None:
    """Salva um resumo simples das métricas no diretório de saída."""
    if summary.get("status") != "ok":
        return

    metrics = {
        "accuracy": utils.to_python_scalar(
            summary.get("final_global_accuracy", 0.0) or 0.0
        ),
        "balanced_accuracy": utils.to_python_scalar(
            summary.get("final_global_balanced_accuracy", 0.0) or 0.0
        ),
        "f1_macro": utils.to_python_scalar(
            summary.get("final_global_f1_macro", 0.0) or 0.0
        ),
        "validation_accuracy": utils.to_python_scalar(
            summary.get("final_validation_accuracy", 0.0) or 0.0
        ),
        "best_validation_accuracy": utils.to_python_scalar(
            summary.get("best_validation_accuracy", 0.0) or 0.0
        ),
        "epsilon": float(summary.get("epsilon") or 0.0),
        "beta": float(config.CVAE_BETA),
        "seed": int(summary.get("seed", 0)),
    }

    # Inclui alpha somente quando Dirichlet; o Flautim recebe apenas escalares.
    if summary.get("dirichlet_alpha") is not None:
        metrics["dirichlet_alpha"] = float(summary["dirichlet_alpha"])

    metrics_path = get_output_root() / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class ServerMetricCollector:
    """Intercepta os callbacks do servidor Flower para registrar métricas completas."""

    def __init__(self, server_module) -> None:
        self.server_module = server_module
        self._original_get_evaluate_fn = server_module.get_evaluate_fn
        self._original_weighted_average = server_module.weighted_average
        self._original_fit_metrics_aggregation_fn = (
            server_module.fit_metrics_aggregation_fn
        )
        self._installed = False
        self.reset()

    def reset(self) -> None:
        self.centralized: list[dict[str, Any]] = []
        self.distributed: list[dict[str, Any]] = []
        self.fit: list[dict[str, Any]] = []
        self._distributed_round = 0
        self._fit_round = 0

    def install(self) -> None:
        if self._installed:
            return

        collector = self

        def recording_get_evaluate_fn(test_loader):
            base_evaluate = collector._original_get_evaluate_fn(test_loader)

            def recording_evaluate(server_round, parameters, config_dict):
                result = base_evaluate(server_round, parameters, config_dict)
                if result is not None:
                    loss, metrics = result
                    row = {
                        "round": int(server_round),
                        "centralized_loss": utils.to_python_scalar(loss),
                    }
                    for name, val in (metrics or {}).items():
                        row[f"centralized_{name}"] = utils.to_python_scalar(val)
                    collector.centralized.append(row)
                return result

            return recording_evaluate

        def recording_weighted_average(metrics):
            aggregated = collector._original_weighted_average(metrics)
            collector._distributed_round += 1
            row = {"round": collector._distributed_round}
            for name, val in (aggregated or {}).items():
                row[f"distributed_{name}"] = utils.to_python_scalar(val)
            collector.distributed.append(row)
            return aggregated

        def recording_fit_metrics_aggregation_fn(metrics):
            aggregated = collector._original_fit_metrics_aggregation_fn(metrics)
            collector._fit_round += 1
            row = {"round": collector._fit_round}
            for name, val in (aggregated or {}).items():
                row[f"fit_{name}"] = utils.to_python_scalar(val)
            collector.fit.append(row)
            return aggregated

        self.server_module.get_evaluate_fn = recording_get_evaluate_fn
        self.server_module.weighted_average = recording_weighted_average
        self.server_module.fit_metrics_aggregation_fn = recording_fit_metrics_aggregation_fn
        self._installed = True

    def dataframe(self, seed: int) -> pd.DataFrame:
        rows: dict[int, dict[str, Any]] = {}

        for group in (self.centralized, self.distributed, self.fit):
            for record in group:
                round_num = int(record["round"])
                rows.setdefault(round_num, {}).update(record)

        df = pd.DataFrame.from_dict(rows, orient="index")
        if df.empty:
            return pd.DataFrame(columns=["seed", "round"])

        df.index.name = "round_index"
        df = df.sort_index().reset_index()
        df = df.rename(columns={"round_index": "round"})
        df.insert(0, "seed", seed)
        return df

    def build_summary(self, seed: int) -> dict[str, Any]:
        if not self.centralized:
            raise RuntimeError("A simulação terminou sem registrar métricas centralizadas.")

        final_centralized = max(self.centralized, key=lambda r: r["round"])
        final_distributed = max(self.distributed, key=lambda r: r["round"]) if self.distributed else None

        best_distributed = None
        if self.distributed:
            acc_rows = [r for r in self.distributed if r.get("distributed_accuracy") is not None]
            if acc_rows:
                best_distributed = max(acc_rows, key=lambda r: float(r["distributed_accuracy"]))

        return {
            "seed": seed,
            "status": "ok",
            "dataset": config.DATASET,
            "extractor": config.EXTRACTOR_NAME,
            "mode": config.TRAIN_MODE,
            "use_dp_cvae": config.USE_LOCAL_DP_CVAE,
            "epsilon": config.TARGET_EPSILON if config.USE_LOCAL_DP_CVAE else None,
            "partitioner": config.PARTITIONER,
            "dirichlet_alpha": config.DIRICHLET_ALPHA if config.PARTITIONER == "dirichlet" else None,
            "server_rounds": config.NUM_SERVER_ROUNDS,
            "beta": float(config.CVAE_BETA),
            # Métricas Limpas Globais
            "final_global_accuracy": final_centralized.get("centralized_global_accuracy"),
            "final_global_balanced_accuracy": final_centralized.get("centralized_global_balanced_accuracy"),
            "final_global_f1_macro": final_centralized.get("centralized_global_f1_macro"),
            # Ruído σ = 0.10
            "acc_noise_008": final_centralized.get("centralized_accuracy_noise_008"),
            "balanced_acc_noise_008": final_centralized.get("centralized_balanced_accuracy_noise_008"),
            "f1_noise_008": final_centralized.get("centralized_f1_noise_008"),
            "retention_acc_008": final_centralized.get("centralized_retention_accuracy_008"),
            "retention_f1_008": final_centralized.get("centralized_retention_f1_008"),
            # Ruído σ = 0.25
            "acc_noise_018": final_centralized.get("centralized_accuracy_noise_018"),
            "balanced_acc_noise_018": final_centralized.get("centralized_balanced_accuracy_noise_018"),
            "f1_noise_018": final_centralized.get("centralized_f1_noise_018"),
            "retention_acc_018": final_centralized.get("centralized_retention_accuracy_018"),
            "retention_f1_018": final_centralized.get("centralized_retention_f1_018"),
            # Ruído σ = 0.50
            "acc_noise_026": final_centralized.get("centralized_accuracy_noise_026"),
            "balanced_acc_noise_026": final_centralized.get("centralized_balanced_accuracy_noise_026"),
            "f1_noise_026": final_centralized.get("centralized_f1_noise_026"),
            "retention_acc_026": final_centralized.get("centralized_retention_accuracy_026"),
            "retention_f1_026": final_centralized.get("centralized_retention_f1_026"),
            # Validação Distribuída
            "final_validation_accuracy": final_distributed.get("distributed_accuracy") if final_distributed else None,
            "best_validation_round": best_distributed.get("round") if best_distributed else None,
            "best_validation_accuracy": best_distributed.get("distributed_accuracy") if best_distributed else None,
        }


def run_single_experiment(args) -> dict[str, Any]:
    # Aplica argumentos dinâmicos no config global, env vars e arquivo ativo
    config.apply_config(
        dataset=args.dataset,
        partitioner=args.partitioner,
        dirichlet_alpha=args.dirichlet_alpha,
        train_mode=args.mode,
        target_epsilon=args.epsilon,
        num_server_rounds=args.num_rounds,
    )

    seed = args.seed
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else get_output_root() / "experiments" / f"run_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    collector = ServerMetricCollector(server)
    collector.install()

    clear_synthetic_cache()
    dataset.reset_federated_dataset()
    utils.seed_all(seed)
    collector.reset()

    # Configuração de recursos Ray / Flower Simulation
    if torch.backends.mps.is_available():
        client_resources = {"num_cpus": 2, "num_gpus": 0.0}
    else:
        client_resources = {"num_cpus": 2, "num_gpus": 1.0 if torch.cuda.is_available() else 0.0}
    backend_config = {"client_resources": client_resources}

    try:
        server_app = ServerApp(server_fn=make_seeded_server_fn(server.server_fn, seed))
        client_app = ClientApp(client_fn=make_seeded_client_fn(client_fn, seed))

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=config.NUM_PARTITIONS,
            backend_config=backend_config,
        )

        history_df = collector.dataframe(seed)
        history_df.to_csv(output_dir / "history.csv", index=False)

        summary = collector.build_summary(seed)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_run_metrics(summary)
        return summary

    except Exception as error:
        error_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        (output_dir / "error.txt").write_text(error_text, encoding="utf-8")
        failure = {"seed": seed, "status": "erro", "dataset": config.DATASET,
                   "mode": config.TRAIN_MODE, "partitioner": config.PARTITIONER,
                   "dirichlet_alpha": config.DIRICHLET_ALPHA if config.PARTITIONER == "dirichlet" else None,
                   "epsilon": config.TARGET_EPSILON if config.USE_LOCAL_DP_CVAE else None,
                   "beta": float(config.CVAE_BETA),
                   "error": str(error)}
        (output_dir / "summary.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        return failure

    finally:
        utils.release_resources()
        config.clear_active_config()


def main():
    args = parse_args()
    summary = run_single_experiment(args)
    if summary.get("status") == "ok":
        acc = summary.get("final_global_accuracy", 0.0) * 100
        bal_acc = summary.get("final_global_balanced_accuracy", 0.0) * 100
        f1 = summary.get("final_global_f1_macro", 0.0) * 100
        print(f"\n[SUCESSO] Run Seed={args.seed} concluído! Acc={acc:.2f}% | Bal Acc={bal_acc:.2f}% | F1={f1:.2f}%")
        sys.exit(0)
    else:
        print(f"\n[FALHA] Run Seed={args.seed} falhou: {summary.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()