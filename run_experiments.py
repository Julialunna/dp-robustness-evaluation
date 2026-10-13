"""
Orquestrador Geral de Experimentos (Grid Runner & Plotter).

Executa automaticamente todas as combinações da grade de pesquisa em subprocessos isolados,
salva os resultados consolidados em CSV e gera gráficos de retenção de Acurácia e F1-Score.

Pode ser executado diretamente via terminal ou dentro de um Jupyter Notebook:
    python run_experiments.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# ==============================================================================
# DEFINIÇÃO DA GRADE DE EXPERIMENTOS
# ==============================================================================
SEEDS = config.GRID_SEEDS  # [2026, 2027, 2028, 2029, 2030]

PARTITIONS = [
    {"name": "iid", "partitioner": "iid", "alpha": 1.0, "label": "IID"},
    {"name": "dirichlet_1.0", "partitioner": "dirichlet", "alpha": 1.0, "label": "Dirichlet (α=1.0)"},
    {"name": "dirichlet_0.5", "partitioner": "dirichlet", "alpha": 0.5, "label": "Dirichlet (α=0.5)"},
]

MODELS_CONFIG = [
    {"mode": "baseline", "epsilon": 0.0, "label": "Baseline (DINO Puro)", "color": "#1f77b4", "marker": "o"},
    {"mode": "cvae_sem_dp", "epsilon": 0.0, "label": "CVAE sem DP", "color": "#2ca02c", "marker": "s"},
    {"mode": "cvae_dp", "epsilon": 3.0, "label": "DP-CVAE (ε=3.0)", "color": "#ff7f0e", "marker": "^"},
    {"mode": "cvae_dp", "epsilon": 5.0, "label": "DP-CVAE (ε=5.0)", "color": "#d62728", "marker": "v"},
    {"mode": "cvae_dp", "epsilon": 8.0, "label": "DP-CVAE (ε=8.0)", "color": "#9467bd", "marker": "d"},
]


def plot_retention_graphs(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Gera gráficos de retenção e qualidade vs nível de ruído para cada particionamento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_levels = [0.0, 0.10, 0.25, 0.50]
    noise_labels = ["0.0 (Limpo)", "0.10", "0.25", "0.50"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for part in PARTITIONS:
        part_name = part["partitioner"]
        alpha_val = part["alpha"]
        part_label = part["label"]

        # Filtra dados para a partição atual
        if part_name == "iid":
            part_df = results_df[results_df["partitioner"] == "iid"]
        else:
            part_df = results_df[
                (results_df["partitioner"] == "dirichlet") & (results_df["dirichlet_alpha"] == alpha_val)
            ]

        if part_df.empty:
            continue

        # ----------------------------------------------------------------------
        # 1. Gráfico de Retenção de Acurácia (%) vs Ruído
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

        for model_cfg in MODELS_CONFIG:
            mode = model_cfg["mode"]
            eps = model_cfg["epsilon"]
            label = model_cfg["label"]
            color = model_cfg["color"]
            marker = model_cfg["marker"]

            if mode == "cvae_dp":
                m_df = part_df[(part_df["mode"] == mode) & (part_df["epsilon"] == eps)]
            else:
                m_df = part_df[part_df["mode"] == mode]

            if m_df.empty:
                continue

            # Retenções médias e desvios para os 4 níveis de ruído
            # σ = 0.0 é sempre 100% de retenção por definição
            ret_means = [100.0]
            ret_stds = [0.0]

            for col in ["retention_acc_010", "retention_acc_025", "retention_acc_050"]:
                ret_means.append(m_df[col].mean())
                ret_stds.append(m_df[col].std())

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

        ax.set_title(f"Retenção de Acurácia sob Ruído Gaussiano — {part_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Desvio Padrão do Ruído (σ)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Retenção de Acurácia (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels(noise_labels)
        ax.set_ylim(bottom=0, top=105)
        ax.legend(frameon=True, fontsize=10, loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.6)

        plot_path = output_dir / f"retencao_acuracia_{part['name']}.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"  📊 Gráfico salvo: {plot_path}")

        # ----------------------------------------------------------------------
        # 2. Gráfico de Retenção de Macro F1-Score (%) vs Ruído
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

        for model_cfg in MODELS_CONFIG:
            mode = model_cfg["mode"]
            eps = model_cfg["epsilon"]
            label = model_cfg["label"]
            color = model_cfg["color"]
            marker = model_cfg["marker"]

            if mode == "cvae_dp":
                m_df = part_df[(part_df["mode"] == mode) & (part_df["epsilon"] == eps)]
            else:
                m_df = part_df[part_df["mode"] == mode]

            if m_df.empty:
                continue

            ret_means = [100.0]
            ret_stds = [0.0]

            for col in ["retention_f1_010", "retention_f1_025", "retention_f1_050"]:
                ret_means.append(m_df[col].mean())
                ret_stds.append(m_df[col].std())

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

        ax.set_title(f"Retenção de F1-Score (Macro) sob Ruído — {part_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Desvio Padrão do Ruído (σ)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Retenção de F1-Score (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels(noise_labels)
        ax.set_ylim(bottom=0, top=105)
        ax.legend(frameon=True, fontsize=10, loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.6)

        plot_path = output_dir / f"retencao_f1_{part['name']}.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"  📊 Gráfico salvo: {plot_path}")

        # ----------------------------------------------------------------------
        # 3. Gráfico de Acurácia Balanceada Absoluta (%) vs Ruído
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

        for model_cfg in MODELS_CONFIG:
            mode = model_cfg["mode"]
            eps = model_cfg["epsilon"]
            label = model_cfg["label"]
            color = model_cfg["color"]
            marker = model_cfg["marker"]

            if mode == "cvae_dp":
                m_df = part_df[(part_df["mode"] == mode) & (part_df["epsilon"] == eps)]
            else:
                m_df = part_df[part_df["mode"] == mode]

            if m_df.empty:
                continue

            bal_means = [m_df["final_global_balanced_accuracy"].mean() * 100]
            bal_stds = [m_df["final_global_balanced_accuracy"].std() * 100]

            for col in ["balanced_acc_noise_010", "balanced_acc_noise_025", "balanced_acc_noise_050"]:
                bal_means.append(m_df[col].mean() * 100)
                bal_stds.append(m_df[col].std() * 100)

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

        ax.set_title(f"Acurácia Balanceada sob Ruído — {part_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Desvio Padrão do Ruído (σ)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Acurácia Balanceada (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(noise_levels)
        ax.set_xticklabels(noise_labels)
        ax.legend(frameon=True, fontsize=10, loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.6)

        plot_path = output_dir / f"acuracia_balanceada_{part['name']}.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"  📊 Gráfico salvo: {plot_path}")


def get_grid_directory(dataset_name: str) -> Path:
    """Retorna a pasta da grade mais recente para permitir continuação (resume) ou cria uma nova."""
    base_parent = Path("artifacts/experiments")
    base_parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [d for d in base_parent.glob(f"grid_{dataset_name}_*") if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in existing:
        if any(d.glob("*/summary.json")):
            print(f"🔄 Retomando execução na pasta existente: {d}\n")
            return d

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = base_parent / f"grid_{dataset_name}_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def run_full_grid():
    base_dir = get_grid_directory(config.DATASET)
    plots_dir = base_dir / "plots"

    total_combinations = len(PARTITIONS) * len(MODELS_CONFIG) * len(SEEDS)
    current_run = 0

    print("=" * 80)
    print("🚀 INICIANDO GRADE COMPLETA DE EXPERIMENTOS")
    print(f"Dataset: {config.DATASET} | Extrator: {config.EXTRACTOR_NAME}")
    print(f"Partições ({len(PARTITIONS)}): {[p['name'] for p in PARTITIONS]}")
    print(f"Modalidades ({len(MODELS_CONFIG)}): {[m['label'] for m in MODELS_CONFIG]}")
    print(f"Sementes ({len(SEEDS)}): {SEEDS}")
    print(f"Total de Execuções: {total_combinations}")
    print(f"Resultados salvos em: {base_dir}")
    print("=" * 80 + "\n")

    all_summaries: List[Dict] = []
    grid_csv_path = base_dir / "grid_results.csv"

    for part in PARTITIONS:
        for model_cfg in MODELS_CONFIG:
            for seed in SEEDS:
                current_run += 1
                cond_name = f"{part['name']}_{model_cfg['mode']}_eps{model_cfg['epsilon']}_seed{seed}"
                run_dir = base_dir / cond_name
                run_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"\n[{current_run}/{total_combinations}] Executando: {part['label']} | "
                    f"{model_cfg['label']} | Seed={seed}"
                )

                # Verifica se a execução já foi concluída com sucesso (Resume automático)
                summary_json = run_dir / "summary.json"
                if summary_json.exists():
                    try:
                        summary_data = json.loads(summary_json.read_text(encoding="utf-8"))
                        if summary_data.get("status") == "ok":
                            acc_pct = summary_data.get("final_global_accuracy", 0.0) * 100
                            print(f"  ⏭️ Já concluído com sucesso (Acc={acc_pct:.2f}%). Pulando...")
                            summary_data["partition_label"] = part["label"]
                            summary_data["model_label"] = model_cfg["label"]
                            all_summaries.append(summary_data)
                            pd.DataFrame(all_summaries).to_csv(grid_csv_path, index=False)
                            continue
                    except Exception:
                        pass

                cmd = [
                    sys.executable,
                    "main.py",
                    "--dataset", config.DATASET,
                    "--partitioner", part["partitioner"],
                    "--dirichlet-alpha", str(part["alpha"]),
                    "--mode", model_cfg["mode"],
                    "--epsilon", str(model_cfg["epsilon"]),
                    "--seed", str(seed),
                    "--num-rounds", str(config.NUM_SERVER_ROUNDS),
                    "--output-dir", str(run_dir),
                ]

                # Executa como subprocesso completamente isolado
                proc = subprocess.run(cmd)

                summary_json = run_dir / "summary.json"
                if summary_json.exists():
                    try:
                        summary_data = json.loads(summary_json.read_text(encoding="utf-8"))
                        summary_data["partition_label"] = part["label"]
                        summary_data["model_label"] = model_cfg["label"]
                        all_summaries.append(summary_data)
                    except Exception as e:
                        print(f"  ⚠️ Erro ao ler summary.json: {e}")
                else:
                    all_summaries.append({
                        "seed": seed,
                        "status": "erro",
                        "partitioner": part["partitioner"],
                        "dirichlet_alpha": part["alpha"],
                        "mode": model_cfg["mode"],
                        "epsilon": model_cfg["epsilon"],
                        "partition_label": part["label"],
                        "model_label": model_cfg["label"],
                    })

                # Salva o CSV incremental após cada execução (para não perder nada se interromper)
                pd.DataFrame(all_summaries).to_csv(grid_csv_path, index=False)

    print("\n" + "=" * 80)
    print("✅ GRADE DE EXPERIMENTOS CONCLUÍDA!")
    print(f"Resultados consolidados em: {grid_csv_path}")
    print("=" * 80)

    # Gera resumo estatístico (Média e Desvio Padrão por condição)
    results_df = pd.DataFrame(all_summaries)
    valid_df = results_df[results_df["status"] == "ok"]

    if not valid_df.empty:
        group_cols = ["partitioner", "dirichlet_alpha", "mode", "epsilon", "partition_label", "model_label"]
        numeric_cols = valid_df.select_dtypes(include=[np.number]).columns.drop("seed", errors="ignore")
        summary_agg = valid_df.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
        summary_csv_path = base_dir / "grid_summary.csv"
        summary_agg.to_csv(summary_csv_path, index=False)
        print(f"📊 Resumo estatístico salvo em: {summary_csv_path}")

        print("\n📈 Gerando gráficos de retenção e acurácia...")
        plot_retention_graphs(valid_df, plots_dir)
        print(f"✨ Gráficos finalizados em: {plots_dir}")


if __name__ == "__main__":
    run_full_grid()
