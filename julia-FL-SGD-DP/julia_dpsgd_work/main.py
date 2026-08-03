from __future__ import annotations

import gc
import json
import random
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

import parameters_federated
import train


# -----------------------------------------------------------------------------
# Configuração do experimento
# -----------------------------------------------------------------------------
NUM_RUNS = 10
BASE_SEED = 2026


# -----------------------------------------------------------------------------
# Aleatoriedade e limpeza
# -----------------------------------------------------------------------------
def seed_main_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_seeded_server_fn(base_server_fn, run_seed: int):
    def seeded_server_fn(context):
        seed_main_process(run_seed)
        return base_server_fn(context)

    return seeded_server_fn


def make_seeded_client_fn(base_client_fn, run_seed: int):
    # Cada ator Ray recebe sua própria cópia desta variável ao desserializar a função.
    worker_seeded = False

    def seeded_client_fn(context):
        nonlocal worker_seeded
        if not worker_seeded:
            partition_id = int(context.node_config["partition-id"])
            seed_main_process(run_seed + 10_000 + partition_id)
            worker_seeded = True
        return base_client_fn(context)

    return seeded_client_fn


def clear_synthetic_cache() -> None:
    synthetic_cache = Path(parameters_federated.SYNTHETIC_CACHE_DIR)
    if synthetic_cache.exists():
        shutil.rmtree(synthetic_cache)
        print(f"Cache sintético apagado: {synthetic_cache}")


def clear_evaluation_cache_once() -> None:
    evaluation_cache = Path("artifacts/test_embeddings")
    if evaluation_cache.exists():
        shutil.rmtree(evaluation_cache)
        print(f"Cache de avaliação apagado: {evaluation_cache}")


def release_resources() -> None:
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


# -----------------------------------------------------------------------------
# Coleta de métricas
# -----------------------------------------------------------------------------
class ServerMetricCollector:
    """
    Captura as métricas produzidas pelos callbacks do server.py.

    Isto é necessário porque flwr.simulation.run_simulation, na API ServerApp,
    executa a simulação mas retorna None; portanto não existe um objeto History
    para ser lido no main.py.
    """

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

        def recording_get_evaluate_fn(testloader):
            base_evaluate = collector._original_get_evaluate_fn(testloader)

            def recording_evaluate(server_round, parameters, config):
                result = base_evaluate(server_round, parameters, config)
                if result is not None:
                    loss, metrics = result
                    row = {
                        "round": int(server_round),
                        "centralized_loss": to_python_scalar(loss),
                    }
                    for name, value in (metrics or {}).items():
                        row[f"centralized_{name}"] = to_python_scalar(value)
                    collector.centralized.append(row)
                return result

            return recording_evaluate

        def recording_weighted_average(metrics):
            aggregated = collector._original_weighted_average(metrics)
            collector._distributed_round += 1
            row = {"round": collector._distributed_round}
            for name, value in (aggregated or {}).items():
                row[f"distributed_{name}"] = to_python_scalar(value)
            collector.distributed.append(row)
            return aggregated

        def recording_fit_metrics_aggregation_fn(metrics):
            aggregated = collector._original_fit_metrics_aggregation_fn(metrics)
            collector._fit_round += 1
            row = {"round": collector._fit_round}
            for name, value in (aggregated or {}).items():
                row[f"fit_{name}"] = to_python_scalar(value)
            collector.fit.append(row)
            return aggregated

        self.server_module.get_evaluate_fn = recording_get_evaluate_fn
        self.server_module.weighted_average = recording_weighted_average
        self.server_module.fit_metrics_aggregation_fn = (
            recording_fit_metrics_aggregation_fn
        )
        self._installed = True

    def dataframe(self, run_number: int, seed: int) -> pd.DataFrame:
        rows: dict[int, dict[str, Any]] = {}

        for group in (self.centralized, self.distributed, self.fit):
            for record in group:
                round_number = int(record["round"])
                rows.setdefault(round_number, {}).update(record)

        dataframe = pd.DataFrame.from_dict(rows, orient="index")
        if dataframe.empty:
            return pd.DataFrame(columns=["run", "seed", "round"])

        dataframe.index.name = "round_index"
        dataframe = dataframe.sort_index().reset_index(drop=True)
        dataframe.insert(0, "seed", seed)
        dataframe.insert(0, "run", run_number)
        return dataframe

    def build_summary(self, run_number: int, seed: int) -> dict[str, Any]:
        if not self.centralized:
            raise RuntimeError(
                "A simulação terminou, mas nenhuma métrica centralizada foi capturada."
            )

        final_centralized = max(self.centralized, key=lambda row: row["round"])

        final_distributed = None
        best_distributed = None
        if self.distributed:
            final_distributed = max(self.distributed, key=lambda row: row["round"])
            accuracy_rows = [
                row
                for row in self.distributed
                if row.get("distributed_accuracy") is not None
            ]
            if accuracy_rows:
                best_distributed = max(
                    accuracy_rows,
                    key=lambda row: float(row["distributed_accuracy"]),
                )

        return {
            "run": run_number,
            "seed": seed,
            "status": "ok",
            "dataset": parameters_federated.DATASET,
            "extractor": parameters_federated.FOUNDATION_MODEL,
            "use_dp_cvae": parameters_federated.USE_LOCAL_DP_CVAE,
            "epsilon": (
                parameters_federated.TARGET_EPSILON
                if parameters_federated.USE_LOCAL_DP_CVAE
                else None
            ),
            "partitioner": parameters_federated.PARTITIONER,
            "dirichlet_alpha": parameters_federated.DIRICHLET_ALPHA,
            "server_rounds": parameters_federated.NUM_SERVER_ROUNDS,
            "local_epochs": parameters_federated.EPOCHS,
            "learning_rate": parameters_federated.LR,
            "final_global_accuracy": final_centralized.get(
                "centralized_global_accuracy"
            ),
            "final_accuracy_noise_01": final_centralized.get(
                "centralized_global_accuracy_noisy_1"
            ),
            "final_accuracy_noise_025": final_centralized.get(
                "centralized_global_accuracy_noisy_2"
            ),
            "final_accuracy_noise_05": final_centralized.get(
                "centralized_global_accuracy_noisy_3"
            ),
            "final_validation_accuracy": (
                final_distributed.get("distributed_accuracy")
                if final_distributed
                else None
            ),
            "best_validation_round": (
                best_distributed.get("round") if best_distributed else None
            ),
            "best_validation_accuracy": (
                best_distributed.get("distributed_accuracy")
                if best_distributed
                else None
            ),
        }


# -----------------------------------------------------------------------------
# Execução
# -----------------------------------------------------------------------------
def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (
        f"{parameters_federated.DATASET}_"
        f"{parameters_federated.FOUNDATION_MODEL}_"
        f"{'dp' if parameters_federated.USE_LOCAL_DP_CVAE else 'sem_dp'}_"
        f"{timestamp}"
    )
    experiment_dir = Path("artifacts/experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        name: value
        for name, value in vars(parameters_federated).items()
        if name.isupper() and isinstance(value, (str, int, float, bool, type(None)))
    }
    configuration.update({"NUM_INDEPENDENT_RUNS": NUM_RUNS, "BASE_SEED": BASE_SEED})
    (experiment_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train.ensure_embedding_extractor_ready()

    # Importamos o módulo inteiro para conseguir envolver seus callbacks e
    # registrar as métricas sem modificar server.py.
    import server
    from client import client_fn

    collector = ServerMetricCollector(server)
    collector.install()

    clear_evaluation_cache_once()

    client_resources = {
        "num_cpus": 2,
        "num_gpus": 1.0 if torch.cuda.is_available() else 0.0,
    }

    summaries: list[dict[str, Any]] = []

    for run_number in range(1, NUM_RUNS + 1):
        seed = BASE_SEED + run_number - 1
        run_dir = experiment_dir / f"run_{run_number:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"INICIANDO REPETIÇÃO {run_number}/{NUM_RUNS} | seed principal={seed}")
        print("=" * 80 + "\n")

        clear_synthetic_cache()
        train.fds = None
        seed_main_process(seed)
        collector.reset()

        try:
            server_app = ServerApp(
                server_fn=make_seeded_server_fn(server.server_fn, seed)
            )
            client_app = ClientApp(
                client_fn=make_seeded_client_fn(client_fn, seed)
            )

            # run_simulation executa a simulação, mas não retorna History nesta API.
            run_simulation(
                server_app=server_app,
                client_app=client_app,
                num_supernodes=parameters_federated.NUM_PARTITIONS,
                backend_config={"client_resources": client_resources},
            )

            run_history = collector.dataframe(run_number, seed)
            run_history.to_csv(run_dir / "history.csv", index=False)

            summary = collector.build_summary(run_number, seed)
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(
                experiment_dir / "summary.csv", index=False
            )

            final_acc = summary["final_global_accuracy"]
            best_acc = summary["best_validation_accuracy"]
            final_text = f"{final_acc * 100:.2f}%" if final_acc is not None else "N/D"
            best_text = f"{best_acc * 100:.2f}%" if best_acc is not None else "N/D"

            print(
                f"\nRepetição {run_number} concluída | "
                f"acc final={final_text} | "
                f"melhor validação={best_text} "
                f"na rodada {summary['best_validation_round']}"
            )

        except Exception as error:
            error_text = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            (run_dir / "error.txt").write_text(error_text, encoding="utf-8")

            summaries.append(
                {
                    "run": run_number,
                    "seed": seed,
                    "status": "erro",
                    "error": str(error),
                }
            )
            pd.DataFrame(summaries).to_csv(
                experiment_dir / "summary.csv", index=False
            )
            print(f"Repetição {run_number} falhou: {error}")

        finally:
            release_resources()

    successful = pd.DataFrame(summaries)
    successful = successful[successful["status"] == "ok"]

    if not successful.empty:
        metric_columns = [
            "final_global_accuracy",
            "final_accuracy_noise_01",
            "final_accuracy_noise_025",
            "final_accuracy_noise_05",
            "final_validation_accuracy",
            "best_validation_accuracy",
        ]
        available_columns = [
            column
            for column in metric_columns
            if column in successful.columns
            and successful[column].notna().any()
        ]

        aggregate = successful[available_columns].agg(["mean", "std", "min", "max"])
        aggregate.to_csv(experiment_dir / "aggregate_statistics.csv")

    print(f"\nExperimento finalizado. Resultados em: {experiment_dir}")


if __name__ == "__main__":
    main()