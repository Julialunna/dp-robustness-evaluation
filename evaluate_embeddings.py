"""Diagnóstico local do CVAE; não treina classificador nem executa FedAvg.

MSE diagnóstica: média por elemento (a loss de treino soma dimensões).
Erro relativo de norma: abs(||recon|| - ||real||) / ||real||, apenas se
||real|| > 1e-8. Desvios/variâncias amostrais usam ddof=1 e exigem n >= 2.
Distância normalizada: distância entre centroides / sqrt(sum(var(real, ddof=1))).
Dispersão <= 1e-8 ou indefinida resulta em NaN.

seed_summary: estatísticas por exemplo agregam com média ponderada e desvio
amostral combinado via momentos; local_mean_* é média de métricas locais,
sem concatenar embeddings entre clientes. grid_summary agrega essas métricas
entre seeds, nunca tratando clientes como repetições independentes.
Estatísticas latentes e geradas do grid usam treino; validação fica nos CSVs
detalhados. Variância posterior = exp(logvar); KL é registrada sem beta.
Covariâncias/correlações latentes são médias/máximos absolutos fora da diagonal.
PCA e momentos não demonstram equivalência de distribuições.
Nenhum cache federado ou sintético é lido, alterado ou removido.

--betas ativa a ablação em beta_ablation, com retomada por protocolo/dados.
Sem --betas preserva o diagnóstico anterior em latent_analysis. Termos da loss
registrados na ablação são médias por exemplo da última época, com reconstrução
somada nas dimensões; a loss total não é comparável isoladamente entre betas.
"""
import sys

if sys.version_info < (3, 10):
    raise RuntimeError("Este diagnóstico e os módulos existentes exigem Python >= 3.10.")

import argparse
import gc
import hashlib
import inspect
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus.validators import ModuleValidator
from sklearn.decomposition import PCA

import config
import dataset
import extractor
import dp_cvae
import utils
from dp_cvae import CVAE, one_hot, reparameterize, loss_function

PARTITIONS = {'iid': ('iid', None), 'dirichlet_1': ('dirichlet', 1.0),
              'dirichlet_05': ('dirichlet', 0.5)}
SEEDS = (2026, 2027, 2028)
EPS = 1e-8
# As métricas latentes/geradas do grid usam TREINO; validação fica nos detalhes.
# _std no grid é sempre dispersão ENTRE SEEDS, nunca entre exemplos/clientes.
GRID_METRICS = {
    'reconstruction_mse_train': 'reconstruction_train_mse_per_element_mean',
    'reconstruction_mse_val': 'reconstruction_validation_mse_per_element_mean',
    'reconstruction_cosine_val': 'reconstruction_validation_cosine_mean',
    'latent_mu_abs': 'latent_train_mu_abs_mean',
    'latent_posterior_variance': 'latent_train_posterior_variance_mean',
    'latent_kl': 'latent_train_kl_mean',
    'latent_aggregated_variance': 'local_mean_latent_train_aggregated_variance_mean',
    'real_norm': 'generated_train_real_norm_mean',
    'posterior_generated_norm': 'generated_train_posterior_norm_mean',
    'prior_generated_norm': 'generated_train_prior_norm_mean',
    'real_prior_centroid_distance': 'local_mean_generated_train_prior_centroid_distance',
    'prior_generated_variance': 'local_mean_generated_train_prior_variance_mean',
}
GRID_COLUMNS = ['partitioner', 'dirichlet_alpha'] + [
    f'{name}_{stat}' for name in GRID_METRICS for stat in ('mean', 'std')
]
DEFAULT_OUTPUT = Path('output/embedding_diagnostics/latent_analysis')
BETAS = [0.01, 0.1, 0.5, 1.0]
BETA_OUTPUT = Path('output/embedding_diagnostics/beta_ablation')
BETA_GRID_COLUMNS = ['beta'] + GRID_COLUMNS  # 27 colunas, conforme lista aprovada.


class IncompatibleDataError(RuntimeError):
    """Impede substituir resultados válidos quando a identidade dos dados mudou."""


def train_full_cvae(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    input_dim: int = config.EMBEDDING_DIM,
    num_classes: int = config.NUM_CLASSES,
    latent_dim: int = config.CVAE_LATENT_DIM,
    hidden_dim: int = config.CVAE_HIDDEN_DIM,
    batch_size: int = config.CVAE_BATCH_SIZE,
    epochs: int = config.CVAE_EPOCHS,
    lr: float = config.CVAE_LR,
    beta: float = config.CVAE_BETA,
    device: torch.device = torch.device("cpu"),
    training_metrics=None,
) -> tuple:
    """Reproduz train_local_decoder_no_dp_cvae exclusivamente para recuperar
    encoder e decoder. Mesma inicialização, validação, loss e passos Adam; sem DP.
    """

    dataset = TensorDataset(
        embeddings.float(),
        labels.long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = CVAE(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    encoder_errors = ModuleValidator.validate(
        model.encoder,
        strict=False,
    )
    if encoder_errors:
        model.encoder = ModuleValidator.fix(model.encoder)

    decoder_errors = ModuleValidator.validate(
        model.decoder,
        strict=False,
    )
    if decoder_errors:
        model.decoder = ModuleValidator.fix(model.decoder)

    model.to(device)

    encoder_optimizer = torch.optim.Adam(
        model.encoder.parameters(),
        lr=lr,
    )
    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=lr,
    )

    final_loss = 0.0

    final_terms = []
    for epoch in range(epochs):
        model.encoder.train()
        model.decoder.train()

        running_loss = 0.0
        running_examples = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            encoder_optimizer.zero_grad(set_to_none=True)
            decoder_optimizer.zero_grad(set_to_none=True)

            c_batch = one_hot(
                y_batch,
                num_classes=num_classes,
            )

            mu, logvar = model.encoder(
                x_batch,
                c_batch,
            )
            z = reparameterize(mu, logvar)
            recon = model.decoder(z, c_batch)

            loss = loss_function(
                recon,
                x_batch,
                mu,
                logvar,
                beta=beta,
            )

            # Instrumentação sem gradientes/RNG: os termos são apenas medidos.
            # A loss usada no backward continua sendo a função original acima.
            if training_metrics is not None and epoch == epochs - 1:
                with torch.no_grad():
                    reconstruction_term = F.mse_loss(recon, x_batch, reduction='none').sum(1)
                    kl_term = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)
                    final_terms.append(torch.stack([
                        reconstruction_term, kl_term, beta * kl_term,
                        reconstruction_term + beta * kl_term,
                    ], dim=1).detach().cpu())

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            batch_n = y_batch.size(0)
            running_loss += loss.item() * batch_n
            running_examples += batch_n

        final_loss = running_loss / max(running_examples, 1)

    model.eval()
    if training_metrics is not None:
        terms = torch.cat(final_terms).double().numpy()
        for index, name in enumerate(('reconstruction_sum', 'kl', 'beta_kl', 'total_loss')):
            training_metrics.update({f'training_final_epoch_{name}_{stat}': value
                                     for stat, value in sample_stats(terms[:, index]).items()})
    return model, float(final_loss)


@torch.no_grad()
def reconstruct(model, x, labels, device):
    model.eval()
    batches = []
    for start in range(0, len(x), config.CVAE_BATCH_SIZE):
        c = one_hot(labels[start:start + config.CVAE_BATCH_SIZE].to(device))
        mu, _ = model.encoder(x[start:start + config.CVAE_BATCH_SIZE].to(device), c)
        batches.append(model.decoder(mu, c).cpu())  # z = mu; sem ruído.
    return torch.cat(batches) if batches else torch.empty_like(x)


def sample_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {'n': len(values), 'mean': values.mean() if len(values) else np.nan,
            'std': values.std(ddof=1) if len(values) >= 2 else np.nan}


def reconstruction_metrics(real, reconstructed, split):
    x, r = real.double(), reconstructed.double()
    norms = x.norm(dim=1)
    valid = norms > EPS
    quantities = {
        'mse_per_element': (x - r).square().mean(dim=1).numpy(),
        'cosine': F.cosine_similarity(x, r, dim=1, eps=EPS).numpy(),
        'relative_norm_error': ((r.norm(dim=1)[valid] - norms[valid]).abs()
                                / norms[valid]).numpy(),
    }
    return {f'reconstruction_{split}_{metric}_{stat}': value
            for metric, values in quantities.items()
            for stat, value in sample_stats(values).items()}


def distribution_metrics(real, reconstructed, synthetic):
    sets = {'real': real.double(), 'reconstructed': reconstructed.double(),
            'synthetic': synthetic.double()}
    row = {}
    for name, x in sets.items():
        row[f'{name}_n'] = len(x)
        stats = sample_stats(x.norm(dim=1).numpy())
        row[f'{name}_l2_mean'] = stats['mean']
        row[f'{name}_l2_std'] = stats['std']
        row[f'{name}_variance_mean'] = (x.var(dim=0, unbiased=True).mean().item()
                                       if len(x) >= 2 else np.nan)
    spread = (real.double().var(dim=0, unbiased=True).sum().sqrt().item()
              if len(real) >= 2 else np.nan)
    row['real_dispersion'] = spread
    for name in ('reconstructed', 'synthetic'):
        x = sets[name]
        distance = ((sets['real'].mean(0) - x.mean(0)).norm().item()
                    if len(real) and len(x) else np.nan)
        row[f'centroid_real_{name}_distance'] = distance
        row[f'centroid_real_{name}_normalized'] = (
            distance / spread if np.isfinite(spread) and spread > EPS else np.nan)
    return row


@torch.no_grad()
def latent_representations(model, real, labels, device, diagnostic_seed):
    """Mesmo decoder/rótulos; seed explícita para cada origem de z.

    A geração do prior reutiliza a função original. Sua permutação de classes
    é desfeita apenas na saída para alinhar os rótulos, sem alterar a geração.
    Estes sorteios ocorrem DEPOIS da geração do diagnóstico anterior.
    """
    model.eval()
    mus, logvars = [], []
    for start in range(0, len(real), config.CVAE_BATCH_SIZE):
        stop = start + config.CVAE_BATCH_SIZE
        c = one_hot(labels[start:stop].to(device))
        mu, logvar = model.encoder(real[start:stop].to(device), c)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    mu, logvar = torch.cat(mus), torch.cat(logvars)
    # Reparametrização original, com seed própria, sem afetar o treino anterior.
    utils.seed_all(diagnostic_seed)
    z = reparameterize(mu, logvar)
    generated = []
    for start in range(0, len(z), config.CVAE_BATCH_SIZE):
        stop = start + config.CVAE_BATCH_SIZE
        generated.append(model.decoder(z[start:stop].to(device),
                                      one_hot(labels[start:stop].to(device))).cpu())
    posterior = torch.cat(generated)
    utils.seed_all(diagnostic_seed)
    prior, prior_labels = dp_cvae.generate_synthetic_embeddings_from_decoder(
        model.decoder, real_labels=labels, device=device)
    if len(prior) != len(real) or not torch.equal(
            torch.bincount(labels, minlength=config.NUM_CLASSES),
            torch.bincount(prior_labels, minlength=config.NUM_CLASSES)):
        raise ValueError('Comparação posterior/prior não preservou classes ou tamanho')
    order = torch.empty(len(labels), dtype=torch.long)
    for cls in range(config.NUM_CLASSES):
        order[torch.where(labels == cls)[0]] = torch.where(prior_labels == cls)[0]
    prior, aligned_labels = prior[order], prior_labels[order]
    if not torch.equal(aligned_labels, labels):
        raise ValueError('Rótulos do posterior e prior não estão alinhados')
    for value in (mu, logvar, z, posterior, prior):
        if not torch.isfinite(value).all():
            raise ValueError('Diagnóstico latente contém valores não finitos')
    return dict(mu=mu, logvar=logvar, z=z, posterior=posterior, prior=prior)


def latent_metrics(mu, logvar, z, split):
    """KL não ponderada por beta; variâncias/covariâncias amostrais.

    posterior_variance resume TODOS os valores exp(logvar), N x dimensão;
    mu_abs e mu_norm resumem valores por exemplo. Dimensões ficam em colunas
    escalares detalhadas. Correlações indefinidas são omitidas, nunca zeradas.
    """
    mu, logvar, z = mu.double(), logvar.double(), z.double()
    variance = logvar.exp()
    kl_dimensions = -.5 * (1 + logvar - mu.square() - variance)
    kl = kl_dimensions.sum(dim=1)
    if not torch.isfinite(variance).all() or not torch.isfinite(kl).all():
        raise ValueError('Variância posterior ou KL não finita')
    prefix = f'latent_{split}_'
    row = {prefix + 'n': len(mu)}
    def record(name, values):
        row.update({prefix + name + '_' + stat: value
                    for stat, value in sample_stats(values.numpy()).items()})
    record('mu_abs', mu.abs().mean(dim=1))
    record('mu_norm', mu.norm(dim=1))
    record('posterior_variance', variance.flatten())
    record('kl', kl)
    record('kl_per_dimension', kl / mu.shape[1])
    row[prefix + 'kl_min'] = kl.min().item() if len(kl) else np.nan
    row[prefix + 'kl_max'] = kl.max().item() if len(kl) else np.nan
    for dim in range(mu.shape[1]):
        record(f'mu_dim_{dim:02d}', mu[:, dim])
        record(f'posterior_variance_dim_{dim:02d}', variance[:, dim])
        record(f'kl_dim_{dim:02d}', kl_dimensions[:, dim])
        row[prefix + f'mu_variance_dim_{dim:02d}'] = (
            mu[:, dim].var(unbiased=True).item() if len(mu) >= 2 else np.nan)
        row[prefix + f'aggregated_mean_dim_{dim:02d}'] = (
            z[:, dim].mean().item() if len(z) else np.nan)
        row[prefix + f'aggregated_variance_dim_{dim:02d}'] = (
            z[:, dim].var(unbiased=True).item() if len(z) >= 2 else np.nan)
    row[prefix + 'mu_variance_mean'] = (
        mu.var(dim=0, unbiased=True).mean().item() if len(mu) >= 2 else np.nan)
    row[prefix + 'aggregated_variance_mean'] = (
        z.var(dim=0, unbiased=True).mean().item() if len(z) >= 2 else np.nan)
    if len(z) >= 2:
        centered = z - z.mean(0)
        covariance = centered.T @ centered / (len(z) - 1)
        off_diagonal = ~torch.eye(z.shape[1], dtype=torch.bool)
        cov_values = covariance[off_diagonal].abs()
        denom = covariance.diag().clamp_min(0).sqrt()
        denom = denom[:, None] * denom[None, :]
        mask = off_diagonal & (denom > EPS)
        correlations = (covariance[mask] / denom[mask]).abs()
    else:
        cov_values = correlations = torch.empty(0)
    for name, values in [('covariance', cov_values), ('correlation', correlations)]:
        row[prefix + f'aggregated_offdiag_{name}_abs_mean'] = (
            values.mean().item() if len(values) else np.nan)
        row[prefix + f'aggregated_offdiag_{name}_abs_max'] = (
            values.max().item() if len(values) else np.nan)
    return row


def generated_metrics(real, representations, split):
    """Normas por exemplo; variâncias/distâncias são métricas locais."""
    prefix = f'generated_{split}_'
    row = {}
    spread = (real.double().var(dim=0, unbiased=True).sum().sqrt().item()
              if len(real) >= 2 else np.nan)
    for name, x in [('real', real), ('posterior', representations['posterior']),
                    ('prior', representations['prior'])]:
        row.update({prefix + name + '_norm_' + stat: value
                    for stat, value in sample_stats(x.double().norm(dim=1).numpy()).items()})
        row[prefix + name + '_variance_mean'] = (
            x.double().var(dim=0, unbiased=True).mean().item() if len(x) >= 2 else np.nan)
        if name != 'real':
            distance = ((real.double().mean(0) - x.double().mean(0)).norm().item()
                        if len(real) and len(x) else np.nan)
            row[prefix + name + '_centroid_distance'] = distance
            row[prefix + name + '_centroid_normalized'] = (
                distance / spread if np.isfinite(spread) and spread > EPS else np.nan)
    return row


def detailed_latent_metrics(real, representations, split):
    row = latent_metrics(representations['mu'], representations['logvar'],
                         representations['z'], split)
    row.update(generated_metrics(real, representations, split))
    return row


def summarize(client_df, ablation=False):
    """Pondera métricas de exemplos; preserva médias locais para distribuições."""
    rows = []
    ids = ['partition', 'partitioner', 'dirichlet_alpha', 'seed']
    if ablation:
        ids.insert(0, 'beta')
    for identity, group in client_df.groupby(ids, dropna=False, sort=False):
        row = dict(zip(ids, identity))
        row.update(dataset=config.DATASET, extractor=config.EXTRACTOR_NAME, mode='cvae_decoder_no_dp')
        valid = group.loc[group.status.eq('ok')]
        row.update(clients_requested=len(group), clients_ok=len(valid),
                   clients_failed=len(group) - len(valid))
        sample_prefixes = [col[:-5] for col in client_df.columns
                           if col.endswith('_mean') and col[:-5] + '_n' in client_df
                           and col[:-5] + '_std' in client_df]
        for prefix in sample_prefixes:
            col = prefix + '_mean'
            if valid.empty:
                row[prefix + '_n'] = 0
                row[col] = row[prefix + '_std'] = np.nan
                continue
            counts = pd.to_numeric(valid[prefix + '_n'], errors='coerce')
            means = pd.to_numeric(valid[col], errors='coerce')
            stds = pd.to_numeric(valid[prefix + '_std'], errors='coerce')
            mask = counts.gt(0) & np.isfinite(means)
            counts, means, stds = counts[mask], means[mask], stds[mask]
            n = counts.sum()
            mean = (counts * means).sum() / n if n else np.nan
            # n=1 contributes zero within-client sum of squares.
            within = ((counts - 1) * stds.pow(2)).where(counts > 1, 0).sum()
            ss = within + (counts * (means - mean).pow(2)).sum()
            row[prefix + '_n'] = n
            row[col] = mean
            complete_std = not ((counts > 1) & stds.isna()).any()
            row[prefix + '_std'] = np.sqrt(ss / (n - 1)) if n > 1 and complete_std else np.nan
        distribution_columns = list(distribution_metrics(
            torch.empty(0, config.EMBEDDING_DIM),
            torch.empty(0, config.EMBEDDING_DIM),
            torch.empty(0, config.EMBEDDING_DIM)))
        sample_columns = {prefix + '_' + stat for prefix in sample_prefixes
                          for stat in ('mean', 'std', 'n')}
        extrema_columns = [col for col in client_df.columns
                           if col.startswith('latent_') and col.endswith(('_kl_min', '_kl_max'))]
        for col in extrema_columns:
            values = pd.to_numeric(valid[col], errors='coerce')
            row[col] = values.min() if col.endswith('_min') else values.max()
        distribution_columns += [col for col in client_df.columns
                                 if col.startswith(('latent_', 'generated_'))
                                 and col not in sample_columns and col not in extrema_columns]
        for col in dict.fromkeys(distribution_columns):
            if col.endswith('_n'):
                continue
            values = pd.to_numeric(valid[col], errors='coerce') if col in valid else pd.Series(dtype=float)
            row['local_mean_' + col] = values.mean()
            row['local_valid_clients_' + col] = values.count()
        rows.append(row)
    seeds = pd.DataFrame(rows)
    grid_rows = []
    for identity, group in seeds.groupby(ids[:-1], dropna=False, sort=False):
        row = dict(zip(ids[:-1], identity))
        for name, column in GRID_METRICS.items():
            valid = group.loc[group.clients_ok.gt(0)]
            values = pd.to_numeric(valid.get(column, pd.Series(dtype=float)), errors='coerce')
            values = values[np.isfinite(values)]
            row[name + '_mean'] = values.mean()
            row[name + '_std'] = values.std(ddof=1)
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows).reindex(columns=BETA_GRID_COLUMNS if ablation else GRID_COLUMNS)
    validate_grid_columns(grid, ablation)
    return seeds, grid


def validate_grid_columns(grid, ablation=False):
    expected = BETA_GRID_COLUMNS if ablation else GRID_COLUMNS
    count = 27 if ablation else 26
    if len(expected) != count or len(set(expected)) != count:
        raise ValueError(f'Schema do grid deve conter as {count} colunas acordadas')
    if list(grid.columns) != expected:
        raise ValueError('Lista ou ordem de colunas inválida no grid latente')


def stratified_indices(labels, seed=42, per_class=150):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    pieces = [rng.choice(np.flatnonzero(labels == c),
                         min(per_class, int((labels == c).sum())), replace=False)
              for c in range(config.NUM_CLASSES) if (labels == c).any()]
    return np.concatenate(pieces) if pieces else np.array([], dtype=int)


def plot_pca(real, reconstructed, synthetic, labels, synthetic_labels, out, partition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pca = PCA(n_components=2, svd_solver='full').fit(real.numpy())
    real_indices = stratified_indices(labels.numpy())
    z_real = pca.transform(real[real_indices].numpy())
    colors = plt.get_cmap('tab10')
    out.mkdir(parents=True, exist_ok=True)
    for name, x, y in [('reconstructed', reconstructed, labels),
                       ('synthetic', synthetic, synthetic_labels)]:
        indices = real_indices if name == 'reconstructed' else stratified_indices(y.numpy())
        z_other = pca.transform(x[indices].numpy())
        fig, ax = plt.subplots(figsize=(10, 7))
        for cls in range(config.NUM_CLASSES):
            a = labels[real_indices].numpy() == cls
            b = y[indices].numpy() == cls
            ax.scatter(z_real[a, 0], z_real[a, 1], color=colors(cls), marker='o', s=9, alpha=.3)
            ax.scatter(z_other[b, 0], z_other[b, 1], color=colors(cls), marker='x', s=13, alpha=.65)
        handles = [Line2D([], [], color=colors(c), marker='o', linestyle='', label=f'Classe {c}')
                   for c in range(config.NUM_CLASSES)]
        handles += [Line2D([], [], color='black', marker=m, linestyle='', label=n)
                    for m, n in [('o', 'Real'), ('x', name)]]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left')
        ratios = pca.explained_variance_ratio_
        ax.set(xlabel=f'PC1 ({ratios[0]:.2%})', ylabel=f'PC2 ({ratios[1]:.2%})',
               title=f'{partition} | cliente 0 | seed 2026\nPCA ajustada apenas nos reais de treino')
        fig.tight_layout()
        fig.savefig(out / f'{partition}_client0_seed2026_real_vs_{name}.png', dpi=150)
        plt.close(fig)
    return pca.explained_variance_ratio_


def fingerprint(x, labels, loader):
    digest = hashlib.sha256()
    for tensor in (x, labels):
        digest.update(tensor.contiguous().numpy().tobytes())
    return str(getattr(loader.dataset, '_fingerprint', 'unavailable')) + ':' + digest.hexdigest()


def write_results(rows, class_rows, out, ablation=False):
    clients = pd.DataFrame(rows)
    seeds, grid = summarize(clients, ablation)
    classes = pd.DataFrame(class_rows)
    if classes.empty:
        classes = pd.DataFrame(columns=['partition', 'seed', 'client', 'class', 'status'])
    for frame, name in [(clients, 'client_metrics'), (classes, 'class_metrics'),
                        (seeds, 'seed_summary'), (grid, 'grid_summary')]:
        if name == 'grid_summary':
            validate_grid_columns(frame, ablation)
        temp = out / (name + '.csv.tmp')
        frame.to_csv(temp, index=False)
        temp.replace(out / (name + '.csv'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', choices=list(PARTITIONS),
                        help='iid, dirichlet_1 (alpha=1.0) ou dirichlet_05 (alpha=0.5)')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--client', type=int, choices=range(config.NUM_PARTITIONS))
    parser.add_argument('--betas', type=float, nargs='+', choices=BETAS,
                        help='Ativa ablação; sem esta opção preserva o diagnóstico anterior.')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.betas is not None and len(set(args.betas)) != len(args.betas):
        parser.error('--betas não aceita valores repetidos')
    if args.output_dir is None:
        args.output_dir = BETA_OUTPUT if args.betas is not None else DEFAULT_OUTPUT
    return args


def validate_config():
    expected = dict(DATASET='pathmnist', EXTRACTOR_NAME='dinov2_s',
                    NUM_PARTITIONS=10, NUM_CLASSES=9, EMBEDDING_DIM=384,
                    CVAE_LATENT_DIM=32, CVAE_EPOCHS=10, CVAE_BATCH_SIZE=256,
                    CVAE_LR=.001, CVAE_BETA=.01)
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f'Protocolo requer config.{key}={value!r}')


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    default=utils.to_python_scalar), encoding='utf-8')
    temporary.replace(path)


def protocol_metadata(device):
    """Identidade conservadora para retomada; não altera config.CVAE_BETA."""
    from importlib.metadata import version
    sources = [Path(__file__), Path(dp_cvae.__file__), Path(config.__file__),
               Path(dataset.__file__), Path(extractor.__file__), Path(utils.__file__)]
    return dict(schema='beta-ablation-v1',
                sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                parameters={name: getattr(config, name) for name in (
                    'DATASET', 'EXTRACTOR_NAME', 'NUM_PARTITIONS', 'NUM_CLASSES',
                    'EMBEDDING_DIM', 'CVAE_LATENT_DIM', 'CVAE_HIDDEN_DIM',
                    'CVAE_EPOCHS', 'CVAE_BATCH_SIZE', 'CVAE_LR', 'BATCH_SIZE')},
                versions={name: version(name) for name in (
                    'torch', 'numpy', 'flwr-datasets', 'datasets', 'opacus', 'scikit-learn')},
                python=list(sys.version_info[:2]), device=str(device),
                data_shuffle_seed=42,
                grid_columns=BETA_GRID_COLUMNS)


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    default=utils.to_python_scalar).encode()).hexdigest()


def run_key(row):
    return float(row['beta']), row['partition'], int(row['seed']), int(row['client'])


def run_file(out, row):
    beta, partition, seed, client = run_key(row)
    return out / f'beta_{str(beta).replace(".", "p")}' / partition / f'seed_{seed}' / f'client_{client}.json'


def valid_saved_result(payload, protocol):
    """Exige identidade, integridade e métricas; arquivo interrompido não basta."""
    if not isinstance(payload, dict):
        return False
    try:
        body = {key: value for key, value in payload.items() if key != 'digest'}
        if payload['digest'] != content_digest(body) or payload['protocol'] != content_digest(protocol):
            return False
        row, classes = payload['row'], payload['classes']
        beta, partition, seed, client = run_key(row)
        if beta not in BETAS or partition not in PARTITIONS or seed not in SEEDS or client not in range(config.NUM_PARTITIONS):
            return False
        partitioner, alpha = PARTITIONS[partition]
        expected = dict(partitioner=partitioner, dirichlet_alpha=alpha,
                        dataset=config.DATASET, extractor=config.EXTRACTOR_NAME,
                        mode='cvae_decoder_no_dp', embedding_dim=config.EMBEDDING_DIM,
                        latent_dim=config.CVAE_LATENT_DIM, epochs=config.CVAE_EPOCHS,
                        batch_size=config.CVAE_BATCH_SIZE, learning_rate=config.CVAE_LR)
        if any(key not in row or row[key] != value for key, value in expected.items()):
            return False
        if row['status'] not in ('ok', 'erro') or len(classes) != config.NUM_CLASSES:
            return False
        if {c['class'] for c in classes} != set(range(config.NUM_CLASSES)):
            return False
        if any(run_key(c) != run_key(row) or c['status'] != row['status'] for c in classes):
            return False
        if row['status'] == 'ok':
            required = [value.removeprefix('local_mean_') for value in GRID_METRICS.values()]
            required += ['final_training_loss', 'training_final_epoch_reconstruction_sum_mean',
                         'training_final_epoch_kl_mean', 'training_final_epoch_beta_kl_mean',
                         'training_final_epoch_total_loss_mean']
            if any(not np.isfinite(float(row.get(name, np.nan))) for name in required):
                return False
            reconstruction = row['training_final_epoch_reconstruction_sum_mean']
            kl = row['training_final_epoch_kl_mean']
            weighted_kl = row['training_final_epoch_beta_kl_mean']
            total = row['training_final_epoch_total_loss_mean']
            if not (np.isclose(weighted_kl, beta * kl, rtol=1e-5, atol=1e-5)
                    and np.isclose(total, reconstruction + weighted_kl, rtol=1e-5, atol=1e-5)
                    and np.isclose(total, row['final_training_loss'], rtol=1e-5, atol=1e-5)):
                return False
            for split, count in [('train', row['n_train']), ('validation', row['n_validation'])]:
                if count <= 0 or row[f'latent_{split}_n'] != count:
                    return False
                if sum(c[f'latent_{split}_n'] for c in classes) != count:
                    return False
            if row['n_synthetic'] != row['n_train']:
                return False
            if not row['train_identity'] or not row['validation_identity']:
                return False
            class_fields = ['real_n', 'real_l2_mean', 'latent_train_kl_mean',
                            'latent_validation_kl_mean', 'generated_train_prior_norm_mean']
            if any(any(name not in c for name in class_fields) for c in classes):
                return False
            if row['diagnostic_seed_train'] != seed + 100000 or row['diagnostic_seed_validation'] != seed + 200000:
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def load_ablation(out, protocol):
    manifest = out / 'protocol.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError('Protocolo/ambiente mudou. Use um novo --output-dir dentro de beta_ablation.')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Diretório não vazio sem protocolo verificável; use outro --output-dir.')
        atomic_json(protocol, manifest)
    saved = {}
    for path in out.glob('beta_*/*/seed_*/client_*.json'):
        try:
            payload = json.loads(path.read_text())
            if not valid_saved_result(payload, protocol) or path != run_file(out, payload['row']):
                raise ValueError('conteúdo incompatível, incompleto ou corrompido')
            saved[run_key(payload['row'])] = payload
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(f'[retomada] {path}: ignorado ({error})', flush=True)
    return saved


def publish_ablation(saved, out):
    if saved:
        write_results([p['row'] for p in saved.values()],
                      [c for p in saved.values() for c in p['classes']], out, ablation=True)


def plot_beta_prior(real, prior, labels, out, partition, beta, train_identity):
    """Projeção persistida para comparar betas em processos diferentes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    projection = out / 'projections' / f'{partition}_client0.npz'
    if projection.exists():
        with np.load(projection, allow_pickle=False) as stored:
            if stored['train_identity'].item() != train_identity:
                raise ValueError('Dados reais da PCA mudaram; use um novo diretório de ablação.')
            components, mean, ratios = (stored[name].copy() for name in ('components', 'mean', 'ratios'))
    else:
        pca = PCA(n_components=2, svd_solver='full').fit(real.numpy())
        components, mean, ratios = pca.components_, pca.mean_, pca.explained_variance_ratio_
        projection.parent.mkdir(parents=True, exist_ok=True)
        temporary = projection.with_suffix('.npz.tmp')
        with temporary.open('wb') as output:
            np.savez(output, components=components, mean=mean, ratios=ratios, train_identity=train_identity)
        temporary.replace(projection)
    indices = stratified_indices(labels.numpy())
    real_z = (real[indices].numpy() - mean) @ components.T
    prior_z = (prior[indices].numpy() - mean) @ components.T
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.get_cmap('tab10')
    for cls in range(config.NUM_CLASSES):
        mask = labels[indices].numpy() == cls
        ax.scatter(real_z[mask, 0], real_z[mask, 1], c=[colors(cls)], marker='o', s=9, alpha=.3)
        ax.scatter(prior_z[mask, 0], prior_z[mask, 1], c=[colors(cls)], marker='x', s=13, alpha=.65)
    handles = [Line2D([], [], color=colors(c), marker='o', linestyle='', label=f'Classe {c}')
               for c in range(config.NUM_CLASSES)]
    handles += [Line2D([], [], color='black', marker=m, linestyle='', label=n)
                for m, n in [('o', 'Real'), ('x', 'Prior N(0,I)')]]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set(xlabel=f'PC1 ({ratios[0]:.2%})', ylabel=f'PC2 ({ratios[1]:.2%})',
           title=f'{partition} | cliente 0 | seed 2026 | beta={beta}')
    folder = out / f'beta_{str(beta).replace(".", "p")}' / 'plots'
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f'{partition}_real_vs_prior.png', dpi=150)
    plt.close(fig)
    return ratios


def main(argv=None):
    args = parse_args(argv)
    validate_config()
    ablation = args.betas is not None
    betas = args.betas if ablation else [config.CVAE_BETA]
    root = BETA_OUTPUT if ablation else DEFAULT_OUTPUT
    out = args.output_dir.resolve()
    if not out.is_relative_to(root.resolve()):
        raise ValueError(f'Use --output-dir dentro de {root}, para isolar este diagnóstico.')
    if not ablation and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Diretório não vazio: {out}. Use --output-dir com um novo diretório.')
    # Alterações somente em memória; nunca salva configuração nem limpa caches.
    config.TRAIN_MODE = 'cvae_decoder_no_dp'
    device = utils.get_device()
    protocol = protocol_metadata(device) if ablation else None
    saved = load_ablation(out, protocol) if ablation else {}
    out.mkdir(parents=True, exist_ok=True)
    partitions = [args.partition] if args.partition else list(PARTITIONS)
    seeds = [args.seed] if args.seed else list(SEEDS)
    clients = [args.client] if args.client is not None else range(config.NUM_PARTITIONS)
    print(f'Diagnóstico: {len(partitions)*len(seeds)*len(clients)*len(betas)} combinações; '
          f'betas={betas}; dispositivo={device}', flush=True)
    print('MSE por elemento; std ddof=1; distância normalizada por sqrt(sum(var(real, ddof=1))).', flush=True)
    print('seed_summary: métricas por exemplo ponderadas; local_mean_* = média de métricas locais. '
          f'Grid: {27 if ablation else 26} colunas, std entre seeds; latentes e gerados resumem treino.', flush=True)
    if ablation:
        print('Termos da loss: médias por exemplo da última época, reconstrução SOMADA nas dimensões. '
              'Loss total entre betas não é critério isolado de qualidade. '
              'Referências antigas sem metadados completos não são importadas.', flush=True)
        publish_ablation(saved, out)
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
    print('Versão/assinaturas instaladas dos particionadores:', flush=True)
    from importlib.metadata import version
    print('flwr-datasets=' + version('flwr-datasets'), flush=True)
    for cls in (FederatedDataset, DirichletPartitioner, IidPartitioner):
        print(f'{cls.__name__}{inspect.signature(cls)}', flush=True)
    print('Cada par de loaders é criado uma vez por cliente/partição e reutilizado nas seeds; '
          'partições fixas por construção. Seeds variam o treinamento do CVAE. '
          'Embeddings reutilizados somente dos mesmos loaders e extrator congelado, com hash registrado.', flush=True)
    rows, class_rows = [], []
    ext = None
    try:
        for partition in partitions:
            config.PARTITIONER, config.DIRICHLET_ALPHA = PARTITIONS[partition]
            dataset.reset_federated_dataset()  # Apenas cache em memória deste processo.
            for client in clients:
                data = None
                extraction_error = None
                try:
                    utils.seed_all(42)
                    train_loader, val_loader = dataset.get_client_loaders(client, config.NUM_PARTITIONS)
                    if ext is None:
                        ext = extractor.get_feature_extractor(config.EXTRACTOR_NAME, device)
                        ext.eval()
                        ext.requires_grad_(False)
                    if ablation:
                        # Ordem de extração independente de efeitos RNG do carregamento do DINO.
                        utils.seed_all(42)
                    x, y = extractor.extract_embeddings_from_loader(ext, train_loader, device)
                    v, vy = extractor.extract_embeddings_from_loader(ext, val_loader, device)
                    if not len(x) or not len(v):
                        raise ValueError('Treino ou validação vazio')
                    train_id, val_id = fingerprint(x, y, train_loader), fingerprint(v, vy, val_loader)
                    if ablation:
                        for payload in saved.values():
                            prior_row = payload['row']
                            if (prior_row['partition'] == partition and prior_row['client'] == client
                                    and prior_row['status'] == 'ok'
                                    and (prior_row['train_identity'] != train_id
                                         or prior_row['validation_identity'] != val_id)):
                                raise IncompatibleDataError('Dados divergentes de resultados salvos; use outro --output-dir.')
                    data = x, y, v, vy
                except IncompatibleDataError:
                    raise
                except Exception:
                    extraction_error = traceback.format_exc()
                finally:
                    train_loader = val_loader = None
                # Mesmos tensores, rótulos e seed inicial para todos os betas.
                for seed, beta in [(seed, beta) for seed in seeds for beta in betas]:
                    identity = dict(partition=partition, partitioner=config.PARTITIONER,
                                    dirichlet_alpha=config.DIRICHLET_ALPHA, seed=seed, client=client,
                                    dataset=config.DATASET, extractor=config.EXTRACTOR_NAME,
                                    mode='cvae_decoder_no_dp', embedding_dim=config.EMBEDDING_DIM,
                                    latent_dim=config.CVAE_LATENT_DIM, epochs=config.CVAE_EPOCHS,
                                    batch_size=config.CVAE_BATCH_SIZE, learning_rate=config.CVAE_LR,
                                    beta=beta)
                    key = run_key(identity)
                    if ablation and extraction_error and key in saved and saved[key]['row']['status'] == 'ok':
                        raise IncompatibleDataError('Não foi possível revalidar os dados. Resultados válidos existentes foram preservados.')
                    if ablation and not extraction_error and key in saved and saved[key]['row']['status'] == 'ok':
                        print(f'[{partition}] cliente={client} seed={seed} beta={beta}: já concluído e validado', flush=True)
                        continue
                    row = dict(identity, status='erro')
                    model = recon = val_recon = synthetic = sy = None
                    latent_splits = {}
                    run_classes = []
                    print(f'[{partition}] cliente={client} seed={seed} beta={beta}', flush=True)
                    try:
                        if extraction_error:
                            raise RuntimeError(extraction_error)
                        x, y, v, vy = data
                        row.update(n_train=len(x), n_validation=len(v), train_identity=train_id,
                                   validation_identity=val_id, partition_fixed_across_seeds=True)
                        utils.seed_all(seed)
                        training_metrics = {} if ablation else None
                        model, loss = train_full_cvae(x, y, device=device, beta=beta,
                                                      training_metrics=training_metrics)
                        if training_metrics is not None:
                            row.update(training_metrics)
                        with torch.no_grad():
                            recon = reconstruct(model, x, y, device)
                            val_recon = reconstruct(model, v, vy, device)
                            synthetic, sy = dp_cvae.generate_synthetic_embeddings_from_decoder(
                                model.decoder, real_labels=y, device=device)
                        if len(synthetic) != len(x) or not torch.equal(
                                torch.bincount(y, minlength=config.NUM_CLASSES),
                                torch.bincount(sy, minlength=config.NUM_CLASSES)):
                            raise ValueError('Geração não preservou número de exemplos/classes')
                        for tensor in (x, v, recon, val_recon, synthetic):
                            if not torch.isfinite(tensor).all():
                                raise ValueError('Embeddings contêm valores não finitos')
                        row.update(final_training_loss=loss, n_synthetic=len(synthetic))
                        row.update(reconstruction_metrics(x, recon, 'train'))
                        row.update(reconstruction_metrics(v, val_recon, 'validation'))
                        row.update(distribution_metrics(x, recon, synthetic))
                        for split, real, labels, offset in [('train', x, y, 100000),
                                                            ('validation', v, vy, 200000)]:
                            diagnostic_seed = seed + offset
                            row['diagnostic_seed_' + split] = diagnostic_seed
                            representations = latent_representations(
                                model, real, labels, device, diagnostic_seed)
                            latent_splits[split] = representations
                            row.update(detailed_latent_metrics(real, representations, split))
                        row['status'] = 'ok'
                        local_classes = []
                        for cls in range(config.NUM_CLASSES):
                            mask, smask = y == cls, sy == cls
                            class_row = dict(identity, status='ok', **{'class': cls},
                                             **distribution_metrics(x[mask], recon[mask], synthetic[smask]))
                            for split, real, labels in [('train', x, y), ('validation', v, vy)]:
                                class_mask = labels == cls
                                selected = {key: value[class_mask]
                                            for key, value in latent_splits[split].items()}
                                class_row.update(detailed_latent_metrics(real[class_mask], selected, split))
                            local_classes.append(class_row)
                        run_classes = local_classes
                        if client == 0 and seed == 2026:
                            try:
                                if ablation:
                                    ratios = plot_beta_prior(x, latent_splits['train']['prior'], y,
                                                             out, partition, beta, train_id)
                                else:
                                    ratios = plot_pca(x, recon, synthetic, y, sy, out / 'plots', partition)
                                row.update(pca_pc1_explained_variance=ratios[0],
                                           pca_pc2_explained_variance=ratios[1])
                            except Exception:
                                row['plot_error'] = traceback.format_exc()
                                print(row['plot_error'], flush=True)
                    except Exception:
                        row['status'] = 'erro'
                        row['error'] = traceback.format_exc()
                        run_classes = []
                        for cls in range(config.NUM_CLASSES):
                            run_classes.append(dict(identity, status='erro', **{'class': cls},
                                                    error=row['error']))
                        print(row['error'], flush=True)
                    finally:
                        model = recon = val_recon = synthetic = sy = None
                        latent_splits.clear()
                        representations = selected = None
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        elif device.type == 'mps':
                            try:
                                torch.mps.empty_cache()
                            except RuntimeError:
                                pass
                    if ablation:
                        payload = dict(protocol=content_digest(protocol), row=row, classes=run_classes)
                        payload['digest'] = content_digest(payload)
                        if row['status'] == 'ok' and not valid_saved_result(payload, protocol):
                            row['status'] = 'erro'
                            row['error'] = 'Diagnóstico incompleto ou inconsistente; não entra nas estatísticas.'
                            for class_row in run_classes:
                                class_row.update(status='erro', error=row['error'])
                            payload['digest'] = content_digest({k: v for k, v in payload.items() if k != 'digest'})
                        atomic_json(payload, run_file(out, row))
                        saved[key] = payload
                        publish_ablation(saved, out)
                    else:
                        rows.append(row)
                        class_rows.extend(run_classes)
                        write_results(rows, class_rows, out)
                data = x = y = v = vy = None
        results = [p['row'] for p in saved.values()] if ablation else rows
        print(f'Concluído: {sum(r["status"] == "ok" for r in results)}/{len(results)} sucessos; {out}', flush=True)
    finally:
        ext = None
        extractor._EXTRACTOR_CACHE.clear()  # Somente modelos em memória deste processo.
        dataset.reset_federated_dataset()
        gc.collect()


if __name__ == '__main__':
    main()
