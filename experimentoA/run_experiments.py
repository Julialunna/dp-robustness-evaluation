"""
Orquestrador Geral de Experimentos.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


SEEDS = [2026]  # Experimento A: diagnóstico com uma única seed


PARTITIONS = [
    {
        "name": "iid",
        "partitioner": "iid",
        "alpha": 1.0,
        "label": "IID",
    },
    {
        "name": "dirichlet_1.0",
        "partitioner": "dirichlet",
        "alpha": 1.0,
        "label": "Dirichlet (α=1.0)",
    },
    {
        "name": "dirichlet_0.5",
        "partitioner": "dirichlet",
        "alpha": 0.5,
        "label": "Dirichlet (α=0.5)",
    },
]


MODELS_CONFIG = [
    {
        "mode": "cvae_decoder_no_dp",
        "epsilon": 0.0,
        "label": "Experimento A — Decoder separado, sem DP (β=0.1)",
        "color": "#2ca02c",
        "marker": "s",
    },
]

FLAUTIM_OUTPUT_DIR = Path("/output")


def ensure_flautim_output_dir() -> Path:
    """Cria e retorna o diretório de saída que o Flautim expõe ao usuário."""
    FLAUTIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return FLAUTIM_OUTPUT_DIR


def write_grid_metrics(all_summaries: List[Dict], total_combinations: int) -> None:
    """Atualiza /output/metrics.json durante e ao final da grade."""
    output_root = ensure_flautim_output_dir()

    ok_rows = [row for row in all_summaries if row.get("status") == "ok"]
    failed_rows = [row for row in all_summaries if row.get("status") == "erro"]

    metrics = {
        "experiment_A": 1,
        "beta": float(config.CVAE_BETA),
        "decoder_dp": 0,
        "completed_runs": int(len(all_summaries)),
        "successful_runs": int(len(ok_rows)),
        "failed_runs": int(len(failed_rows)),
        "total_runs": int(total_combinations),
    }

    if ok_rows:
        def mean_metric(name: str) -> float:
            values = [
                float(row[name])
                for row in ok_rows
                if row.get(name) is not None
            ]
            return float(np.mean(values)) if values else 0.0

        metrics.update(
            {
                "mean_accuracy": mean_metric("final_global_accuracy"),
                "mean_balanced_accuracy": mean_metric(
                    "final_global_balanced_accuracy"
                ),
                "mean_f1_macro": mean_metric("final_global_f1_macro"),
                "best_accuracy": float(
                    max(
                        row.get("final_global_accuracy", 0.0) or 0.0
                        for row in ok_rows
                    )
                ),
            }
        )

    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def mirror_grid_csv(csv_path: Path) -> None:
    """Mantém uma cópia fácil de baixar diretamente em /output/."""
    if csv_path.exists():
        shutil.copy2(csv_path, ensure_flautim_output_dir() / csv_path.name)





def plot_retention_graphs(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    noise_levels = [
        0.0,
        0.10,
        0.25,
        0.50,
    ]

    noise_labels = [
        "0.0 (Limpo)",
        "0.10",
        "0.25",
        "0.50",
    ]

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid"
        in plt.style.available
        else "default"
    )

    for part in PARTITIONS:
        part_name = (
            part["partitioner"]
        )

        alpha_val = (
            part["alpha"]
        )

        part_label = (
            part["label"]
        )

        if part_name == "iid":
            part_df = results_df[
                results_df[
                    "partitioner"
                ] == "iid"
            ]

        else:
            part_df = results_df[
                (
                    results_df[
                        "partitioner"
                    ]
                    == "dirichlet"
                )
                &
                (
                    results_df[
                        "dirichlet_alpha"
                    ]
                    == alpha_val
                )
            ]

        if part_df.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(8, 5.5),
            dpi=300,
        )

        for model_cfg in MODELS_CONFIG:
            mode = model_cfg[
                "mode"
            ]

            eps = model_cfg[
                "epsilon"
            ]

            label = model_cfg[
                "label"
            ]

            color = model_cfg[
                "color"
            ]

            marker = model_cfg[
                "marker"
            ]

            m_df = part_df[
                (
                    part_df[
                        "mode"
                    ]
                    == mode
                )
                &
                (
                    part_df[
                        "epsilon"
                    ]
                    == eps
                )
            ]

            if m_df.empty:
                continue

            ret_means = [
                100.0
            ]

            ret_stds = [
                0.0
            ]

            for col in [
                "retention_acc_010",
                "retention_acc_025",
                "retention_acc_050",
            ]:
                ret_means.append(
                    m_df[col].mean()
                )

                ret_stds.append(
                    m_df[col].std()
                )

            ax.errorbar(
                noise_levels,
                ret_means,
                yerr=ret_stds,
                label=label,
                color=color,
                marker=marker,
                capsize=4,
                linewidth=2,
                markersize=6,
            )

        ax.set_title(
            (
                "Retenção de Acurácia "
                "sob Ruído Gaussiano — "
                f"{part_label}"
            ),
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        ax.set_xlabel(
            "Desvio Padrão do Ruído (σ)",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Retenção de Acurácia (%)",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xticks(
            noise_levels
        )

        ax.set_xticklabels(
            noise_labels
        )

        ax.set_ylim(
            bottom=0,
            top=105,
        )

        ax.legend(
            frameon=True,
            fontsize=10,
            loc="lower left",
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.6,
        )

        plot_path = (
            output_dir
            / (
                "retencao_acuracia_"
                f"{part['name']}.png"
            )
        )

        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

        print(
            f"  📊 Gráfico salvo: "
            f"{plot_path}"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5.5),
            dpi=300,
        )

        for model_cfg in MODELS_CONFIG:
            mode = model_cfg[
                "mode"
            ]

            eps = model_cfg[
                "epsilon"
            ]

            label = model_cfg[
                "label"
            ]

            color = model_cfg[
                "color"
            ]

            marker = model_cfg[
                "marker"
            ]

            m_df = part_df[
                (
                    part_df[
                        "mode"
                    ]
                    == mode
                )
                &
                (
                    part_df[
                        "epsilon"
                    ]
                    == eps
                )
            ]

            if m_df.empty:
                continue

            ret_means = [
                100.0
            ]

            ret_stds = [
                0.0
            ]

            for col in [
                "retention_f1_010",
                "retention_f1_025",
                "retention_f1_050",
            ]:
                ret_means.append(
                    m_df[col].mean()
                )

                ret_stds.append(
                    m_df[col].std()
                )

            ax.errorbar(
                noise_levels,
                ret_means,
                yerr=ret_stds,
                label=label,
                color=color,
                marker=marker,
                capsize=4,
                linewidth=2,
                markersize=6,
            )

        ax.set_title(
            (
                "Retenção de F1-Score "
                "(Macro) sob Ruído — "
                f"{part_label}"
            ),
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        ax.set_xlabel(
            "Desvio Padrão do Ruído (σ)",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Retenção de F1-Score (%)",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xticks(
            noise_levels
        )

        ax.set_xticklabels(
            noise_labels
        )

        ax.set_ylim(
            bottom=0,
            top=105,
        )

        ax.legend(
            frameon=True,
            fontsize=10,
            loc="lower left",
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.6,
        )

        plot_path = (
            output_dir
            / (
                "retencao_f1_"
                f"{part['name']}.png"
            )
        )

        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

        print(
            f"  📊 Gráfico salvo: "
            f"{plot_path}"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5.5),
            dpi=300,
        )

        for model_cfg in MODELS_CONFIG:
            mode = model_cfg[
                "mode"
            ]

            eps = model_cfg[
                "epsilon"
            ]

            label = model_cfg[
                "label"
            ]

            color = model_cfg[
                "color"
            ]

            marker = model_cfg[
                "marker"
            ]

            m_df = part_df[
                (
                    part_df[
                        "mode"
                    ]
                    == mode
                )
                &
                (
                    part_df[
                        "epsilon"
                    ]
                    == eps
                )
            ]

            if m_df.empty:
                continue

            bal_means = [
                (
                    m_df[
                        "final_global_balanced_accuracy"
                    ].mean()
                    * 100
                )
            ]

            bal_stds = [
                (
                    m_df[
                        "final_global_balanced_accuracy"
                    ].std()
                    * 100
                )
            ]

            for col in [
                "balanced_acc_noise_010",
                "balanced_acc_noise_025",
                "balanced_acc_noise_050",
            ]:
                bal_means.append(
                    m_df[col].mean()
                    * 100
                )

                bal_stds.append(
                    m_df[col].std()
                    * 100
                )

            ax.errorbar(
                noise_levels,
                bal_means,
                yerr=bal_stds,
                label=label,
                color=color,
                marker=marker,
                capsize=4,
                linewidth=2,
                markersize=6,
            )

        ax.set_title(
            (
                "Acurácia Balanceada "
                "sob Ruído — "
                f"{part_label}"
            ),
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        ax.set_xlabel(
            "Desvio Padrão do Ruído (σ)",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_ylabel(
            "Acurácia Balanceada (%)",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xticks(
            noise_levels
        )

        ax.set_xticklabels(
            noise_labels
        )

        ax.legend(
            frameon=True,
            fontsize=10,
            loc="lower left",
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.6,
        )

        plot_path = (
            output_dir
            / (
                "acuracia_balanceada_"
                f"{part['name']}.png"
            )
        )

        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

        print(
            f"  📊 Gráfico salvo: "
            f"{plot_path}"
        )


def get_grid_directory(
    dataset_name: str
) -> Path:

    base_parent = (
        ensure_flautim_output_dir()
        / "experiments"
    )

    base_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = sorted(
        [
            d
            for d
            in base_parent.glob(
                f"grid_{dataset_name}_*"
            )
            if d.is_dir()
        ],
        key=lambda p:
            p.stat().st_mtime,
        reverse=True,
    )

    for d in existing:
        if any(
            d.glob(
                "*/summary.json"
            )
        ):
            print(
                "🔄 Retomando execução "
                f"na pasta existente: {d}\n"
            )

            return d

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    new_dir = (
        base_parent
        / (
            f"grid_{dataset_name}_"
            f"{timestamp}"
        )
    )

    new_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return new_dir


def run_full_grid():
    base_dir = get_grid_directory(
        config.DATASET
    )

    plots_dir = (
        base_dir
        / "plots"
    )

    total_combinations = (
        len(PARTITIONS)
        * len(MODELS_CONFIG)
        * len(SEEDS)
    )

    current_run = 0

    print("=" * 80)
    print(
        "🧪 INICIANDO EXPERIMENTO A "
        "(decoder separado, sem DP, β=0.1)"
    )

    print(
        f"Dataset: {config.DATASET} | "
        f"Extrator: "
        f"{config.EXTRACTOR_NAME}"
    )

    print(
        f"Partições "
        f"({len(PARTITIONS)}): "
        f"{[p['name'] for p in PARTITIONS]}"
    )

    print(
        f"Modalidades "
        f"({len(MODELS_CONFIG)}): "
        f"{[m['label'] for m in MODELS_CONFIG]}"
    )

    print(
        f"Sementes "
        f"({len(SEEDS)}): "
        f"{SEEDS}"
    )

    print(
        f"Total de Execuções: "
        f"{total_combinations}"
    )

    print(
        f"Resultados salvos em: "
        f"{base_dir}"
    )

    print(
        "=" * 80
        + "\n"
    )

    all_summaries: List[
        Dict
    ] = []

    grid_csv_path = (
        base_dir
        / "grid_results.csv"
    )

    for part in PARTITIONS:
        for model_cfg in MODELS_CONFIG:
            for seed in SEEDS:

                current_run += 1

                cond_name = (
                    f"{part['name']}_"
                    f"{model_cfg['mode']}_"
                    f"eps"
                    f"{model_cfg['epsilon']}_"
                    f"seed{seed}"
                )

                run_dir = (
                    base_dir
                    / cond_name
                )

                run_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                print(
                    f"\n[{current_run}/"
                    f"{total_combinations}] "
                    f"Executando: "
                    f"{part['label']} | "
                    f"{model_cfg['label']} | "
                    f"Seed={seed}"
                )

                summary_json = (
                    run_dir
                    / "summary.json"
                )

                if summary_json.exists():
                    try:
                        summary_data = (
                            json.loads(
                                summary_json
                                .read_text(
                                    encoding="utf-8"
                                )
                            )
                        )

                        if (
                            summary_data.get(
                                "status"
                            )
                            == "ok"
                        ):
                            acc_pct = (
                                summary_data.get(
                                    "final_global_accuracy",
                                    0.0,
                                )
                                * 100
                            )

                            print(
                                "  ⏭️ Já concluído "
                                "com sucesso "
                                f"(Acc={acc_pct:.2f}%). "
                                "Pulando..."
                            )

                            summary_data[
                                "partition_label"
                            ] = part[
                                "label"
                            ]

                            summary_data[
                                "model_label"
                            ] = model_cfg[
                                "label"
                            ]

                            all_summaries.append(
                                summary_data
                            )

                            (
                                pd.DataFrame(
                                    all_summaries
                                )
                                .to_csv(
                                    grid_csv_path,
                                    index=False,
                                )
                            )
                            mirror_grid_csv(grid_csv_path)
                            write_grid_metrics(
                                all_summaries,
                                total_combinations,
                            )

                            continue

                    except Exception:
                        pass

                cmd = [
                    sys.executable,
                    "main.py",

                    "--dataset",
                    config.DATASET,

                    "--partitioner",
                    part["partitioner"],

                    "--dirichlet-alpha",
                    str(part["alpha"]),

                    "--mode",
                    model_cfg["mode"],

                    "--epsilon",
                    str(
                        model_cfg[
                            "epsilon"
                        ]
                    ),

                    "--seed",
                    str(seed),

                    "--num-rounds",
                    str(
                        config
                        .NUM_SERVER_ROUNDS
                    ),

                    "--output-dir",
                    str(run_dir),
                ]

                subprocess.run(
                    cmd
                )

                summary_json = (
                    run_dir
                    / "summary.json"
                )

                if summary_json.exists():
                    try:
                        summary_data = (
                            json.loads(
                                summary_json
                                .read_text(
                                    encoding="utf-8"
                                )
                            )
                        )

                        summary_data[
                            "partition_label"
                        ] = part[
                            "label"
                        ]

                        summary_data[
                            "model_label"
                        ] = model_cfg[
                            "label"
                        ]

                        all_summaries.append(
                            summary_data
                        )

                    except Exception as e:
                        print(
                            "  ⚠️ Erro ao ler "
                            f"summary.json: {e}"
                        )

                else:
                    all_summaries.append(
                        {
                            "seed":
                                seed,

                            "status":
                                "erro",

                            "partitioner":
                                part[
                                    "partitioner"
                                ],

                            "dirichlet_alpha":
                                part[
                                    "alpha"
                                ],

                            "mode":
                                model_cfg[
                                    "mode"
                                ],

                            "epsilon":
                                model_cfg[
                                    "epsilon"
                                ],

                            "partition_label":
                                part[
                                    "label"
                                ],

                            "model_label":
                                model_cfg[
                                    "label"
                                ],
                        }
                    )

                (
                    pd.DataFrame(
                        all_summaries
                    )
                    .to_csv(
                        grid_csv_path,
                        index=False,
                    )
                )
                mirror_grid_csv(grid_csv_path)
                write_grid_metrics(
                    all_summaries,
                    total_combinations,
                )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "✅ GRADE DE EXPERIMENTOS "
        "CONCLUÍDA!"
    )

    print(
        "Resultados consolidados em: "
        f"{grid_csv_path}"
    )

    print(
        "=" * 80
    )

    results_df = pd.DataFrame(
        all_summaries
    )

    valid_df = results_df[
        results_df[
            "status"
        ] == "ok"
    ]

    # Garante que o painel do Flautim tenha métricas mesmo se a grade falhar.
    write_grid_metrics(
        all_summaries,
        total_combinations,
    )
    mirror_grid_csv(grid_csv_path)

    if not valid_df.empty:
        group_cols = [
            "partitioner",
            "dirichlet_alpha",
            "mode",
            "epsilon",
            "partition_label",
            "model_label",
        ]

        numeric_cols = (
            valid_df
            .select_dtypes(
                include=[np.number]
            )
            .columns
            .drop(
                "seed",
                errors="ignore",
            )
        )

        summary_agg = (
            valid_df
            .groupby(
                group_cols
            )[numeric_cols]
            .agg(
                [
                    "mean",
                    "std",
                ]
            )
            .reset_index()
        )

        summary_csv_path = (
            base_dir
            / "grid_summary.csv"
        )

        summary_agg.to_csv(
            summary_csv_path,
            index=False,
        )
        mirror_grid_csv(summary_csv_path)
        write_grid_metrics(
            all_summaries,
            total_combinations,
        )

        print(
            "📊 Resumo estatístico "
            f"salvo em: "
            f"{summary_csv_path}"
        )

        print(
            "\n📈 Gerando gráficos "
            "de retenção e acurácia..."
        )

        plot_retention_graphs(
            valid_df,
            plots_dir,
        )

        print(
            "✨ Gráficos finalizados "
            f"em: {plots_dir}"
        )


if __name__ == "__main__":
    run_full_grid()