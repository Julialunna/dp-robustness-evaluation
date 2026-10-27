"""
Módulo do Cliente Federado (Flower Client).

Cada cliente é responsável por:
1. Extrair os embeddings das suas imagens locais usando o extrator DINOv2 oficial.
2. Treinar no modo configurado:
   - "baseline": Treina o classificador diretamente nos embeddings reais do DINO.
   - "cvae_sem_dp" / "cvae_dp": Treina um modelo CVAE local e gera embeddings sintéticos.
3. Avaliar o classificador nas suas amostras locais de teste (Acurácia, Acurácia Balanceada, F1-Score).
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

import config
import dataset
import dp_cvae
import extractor
import models
import utils

warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    """Cliente Federado Flower com suporte a Baseline, CVAE e DP-CVAE."""

    def __init__(
        self,
        train_loader,
        test_loader,
        partition_id: int,
        context: Context,
    ) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.partition_id = int(partition_id)
        self.context = context
        self.device = utils.get_device()

        # Extrator carregado sob demanda (lazy) apenas se necessário extrair embeddings de imagens
        self._extractor = None

        # Classificador downstream que opera sobre os vetores de embedding
        self.model = models.EmbeddingClassifier(
            input_dim=config.EMBEDDING_DIM,
            hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            dropout=config.CLASSIFIER_DROPOUT,
        )

        mode_desc = f"{config.TRAIN_MODE}"
        if config.USE_LOCAL_DP_CVAE:
            mode_desc += f" (ε={config.TARGET_EPSILON})"
        print(
            f"[Cliente {self.partition_id}] Conectado | Modo={mode_desc} | "
            f"Partição={config.PARTITIONER} (α={config.DIRICHLET_ALPHA})"
        )

    @property
    def extractor(self):
        """Instancia o extrator DINOv2 sob demanda via singleton apenas se necessário."""
        if self._extractor is None:
            self._extractor = extractor.get_feature_extractor(
                name=config.EXTRACTOR_NAME,
                device=self.device,
            )
        return self._extractor

    def get_parameters(self, config_dict: dict):
        return models.get_weights(self.model)

    def _train_embedding_cache_path(self) -> Path:
        alpha_tag = f"alpha_{config.DIRICHLET_ALPHA}" if config.PARTITIONER == "dirichlet" else "iid"
        return (
            Path(config.REAL_EMBEDDINGS_CACHE_DIR)
            / f"client_{self.partition_id}_{config.DATASET}_{config.PARTITIONER}_{alpha_tag}_seed_{config.CURRENT_SEED}_train.pt"
        )

    def _synthetic_cache_path(self) -> Path:
        mode_tag = config.TRAIN_MODE
        eps_tag = f"eps_{config.TARGET_EPSILON}" if config.USE_LOCAL_DP_CVAE else "none"
        alpha_tag = f"alpha_{config.DIRICHLET_ALPHA}" if config.PARTITIONER == "dirichlet" else "iid"
        return (
            Path(config.SYNTHETIC_CACHE_DIR)
            / f"client_{self.partition_id}_{config.DATASET}_{config.PARTITIONER}_{alpha_tag}_seed_{config.CURRENT_SEED}_{mode_tag}_{eps_tag}_synthetic.pt"
        )

    def _test_embedding_cache_path(self) -> Path:
        alpha_tag = f"alpha_{config.DIRICHLET_ALPHA}" if config.PARTITIONER == "dirichlet" else "iid"
        return (
            Path(config.TEST_EMBEDDINGS_CACHE_DIR)
            / f"client_{self.partition_id}_{config.DATASET}_{config.PARTITIONER}_{alpha_tag}_seed_{config.CURRENT_SEED}_test.pt"
        )

    def _extractor_signature(self) -> dict:
        return {
            "extractor_name": config.EXTRACTOR_NAME,
            "image_size": config.IMAGE_SIZE,
            "embedding_dim": self.extractor.embedding_dim,
        }

    def _train_embedding_metadata(self) -> dict:
        metadata = {
            "cache_version": 3,
            "seed": config.CURRENT_SEED,
            "partition_id": self.partition_id,
            "num_train_examples": len(self.train_loader.dataset),
            "partitioner": config.PARTITIONER,
            "dirichlet_alpha": config.DIRICHLET_ALPHA,
            "dataset": config.DATASET,
            "num_classes": config.NUM_CLASSES,
        }
        metadata.update(self._extractor_signature())
        return metadata

    def _expected_cache_metadata(self) -> dict:
        metadata = {
            "cache_version": config.SYNTHETIC_CACHE_VERSION,
            "seed": config.CURRENT_SEED,
            "dataset": config.DATASET,
            "num_classes": config.NUM_CLASSES,
            "partition_id": self.partition_id,
            "num_real_train_examples": len(self.train_loader.dataset),
            "partitioner": config.PARTITIONER,
            "dirichlet_alpha": config.DIRICHLET_ALPHA,
            "train_mode": config.TRAIN_MODE,
            "use_local_dp_cvae": config.USE_LOCAL_DP_CVAE,
            "use_decoder_only_dp": config.USE_DECODER_ONLY_DP,
            "use_full_cvae_dp": config.USE_FULL_CVAE_DP,
            "target_epsilon": config.TARGET_EPSILON if config.USE_LOCAL_DP_CVAE else None,
            "target_delta": config.TARGET_DELTA,
            "max_grad_norm": config.MAX_GRAD_NORM,
            "cvae_hidden_dim": config.CVAE_HIDDEN_DIM,
            "cvae_latent_dim": config.CVAE_LATENT_DIM,
            "cvae_batch_size": config.CVAE_BATCH_SIZE,
            "cvae_epochs": config.CVAE_EPOCHS,
            "cvae_lr": config.CVAE_LR,
            "cvae_beta": config.CVAE_BETA,
            "normalize_embeddings": config.NORMALIZE_EMBEDDINGS,
        }
        metadata.update(self._extractor_signature())
        return metadata

    def _test_embedding_metadata(self) -> dict:
        metadata = {
            "cache_version": 3,
            "seed": config.CURRENT_SEED,
            "partition_id": self.partition_id,
            "num_test_examples": len(self.test_loader.dataset),
            "partitioner": config.PARTITIONER,
            "dirichlet_alpha": config.DIRICHLET_ALPHA,
            "dataset": config.DATASET,
            "num_classes": config.NUM_CLASSES,
        }
        metadata.update(self._extractor_signature())
        return metadata

    @staticmethod
    def _cache_is_valid(cached_metadata: dict, expected_metadata: dict) -> bool:
        for key, expected_val in expected_metadata.items():
            if cached_metadata.get(key) != expected_val:
                return False
        return True

    @staticmethod
    def _normalize_embeddings(
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padroniza os embeddings reais localmente por dimensão: (x - mean) / std."""
        mean = embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, keepdim=True, unbiased=False).clamp_min(
            config.EMBEDDING_NORM_EPS
        )
        normalized = (embeddings - mean) / std
        return normalized, mean, std

    @staticmethod
    def _denormalize_embeddings(
        embeddings: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Restaura a escala original dos embeddings sintéticos: x_norm * std + mean."""
        return embeddings * std + mean

    def _extract_or_load_real_train_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carrega do cache ou extrai os embeddings reais do DINOv2 para a partição de treino local."""
        cache_path = self._train_embedding_cache_path()
        expected_metadata = self._train_embedding_metadata()

        if cache_path.exists():
            try:
                cached_data = torch.load(cache_path, map_location="cpu")
                if self._cache_is_valid(cached_data.get("metadata", {}), expected_metadata):
                    return cached_data["embeddings"].float(), cached_data["labels"].long()
            except Exception:
                pass

        print(f"[Cliente {self.partition_id}] Extraindo embeddings reais locais via {config.EXTRACTOR_NAME}...")
        real_embeddings, real_labels = extractor.extract_embeddings_from_loader(
            self.extractor,
            self.train_loader,
            self.device,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": real_embeddings.cpu(),
                "labels": real_labels.cpu(),
                "metadata": expected_metadata,
            },
            cache_path,
        )
        return real_embeddings.float(), real_labels.long()

    def _load_baseline_loader(self):
        """Modo Baseline: Carrega ou extrai os embeddings reais do DINO para o train_loader."""
        real_embeddings, real_labels = self._extract_or_load_real_train_embeddings()
        loader = models.make_embedding_loader(
            real_embeddings,
            real_labels,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
        )
        return loader, None, None, len(real_labels)

    def _build_or_load_synthetic_loader(self):
        """Modo CVAE / DP-CVAE: Carrega dados sintéticos do cache ou treina o CVAE local para gerá-los."""
        cache_path = self._synthetic_cache_path()
        should_retrain = config.RETRAIN_CVAE_EVERY_ROUND or config.FORCE_REGENERATE_SYNTHETICS
        expected_metadata = self._expected_cache_metadata()

        if cache_path.exists() and not should_retrain:
            try:
                synthetic_embeddings, synthetic_labels, metadata = dp_cvae.load_synthetic_dataset(cache_path)
                if self._cache_is_valid(metadata, expected_metadata):
                    epsilon = metadata.get("epsilon")
                    cvae_loss = metadata.get("cvae_final_loss")
                    print(
                        f"[Cliente {self.partition_id}] Sintéticos carregados do cache "
                        f"({len(synthetic_labels)} amostras)."
                    )
                    synthetic_loader = models.make_embedding_loader(
                        synthetic_embeddings,
                        synthetic_labels,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                    )
                    return synthetic_loader, epsilon, cvae_loss, len(synthetic_labels)
            except Exception:
                pass  # Regenera se houver erro ao carregar

            print(
                f"[Cliente {self.partition_id}] Cache sintético incompatível. "
                "Regenerando com a configuração atual."
            )

        real_embeddings, real_labels = self._extract_or_load_real_train_embeddings()

        if config.NORMALIZE_EMBEDDINGS:
            cvae_train_embeddings, emb_mean, emb_std = self._normalize_embeddings(real_embeddings)
        else:
            cvae_train_embeddings = real_embeddings
            emb_mean, emb_std = None, None

        if config.TRAIN_MODE == "cvae_decoder_no_dp":
            dp_status = "sem DP, com encoder/decoder em otimizadores separados"
        elif config.TRAIN_MODE == "cvae_decoder_dp":
            dp_status = f"com DP-SGD somente no decoder (ε={config.TARGET_EPSILON})"
        elif config.TRAIN_MODE == "cvae_dp":
            dp_status = f"com DP-SGD no CVAE inteiro (ε={config.TARGET_EPSILON})"
        else:
            dp_status = "sem DP"
        print(f"[Cliente {self.partition_id}] Treinando CVAE local {dp_status}...")

        if config.TRAIN_MODE == "cvae_decoder_no_dp":
            result = dp_cvae.train_local_decoder_no_dp_cvae(
                cvae_train_embeddings,
                real_labels,
                input_dim=self.extractor.embedding_dim,
                num_classes=config.NUM_CLASSES,
                latent_dim=config.CVAE_LATENT_DIM,
                hidden_dim=config.CVAE_HIDDEN_DIM,
                batch_size=config.CVAE_BATCH_SIZE,
                epochs=config.CVAE_EPOCHS,
                lr=config.CVAE_LR,
                beta=config.CVAE_BETA,
                device=self.device,
            )
            synthetic_embeddings, synthetic_labels = (
                dp_cvae.generate_synthetic_embeddings_from_decoder(
                    result.decoder,
                    real_labels=real_labels,
                    num_classes=config.NUM_CLASSES,
                    latent_dim=config.CVAE_LATENT_DIM,
                    device=self.device,
                )
            )
        elif config.TRAIN_MODE == "cvae_decoder_dp":
            result = dp_cvae.train_local_decoder_dp_cvae(
                cvae_train_embeddings,
                real_labels,
                input_dim=self.extractor.embedding_dim,
                num_classes=config.NUM_CLASSES,
                latent_dim=config.CVAE_LATENT_DIM,
                hidden_dim=config.CVAE_HIDDEN_DIM,
                batch_size=config.CVAE_BATCH_SIZE,
                epochs=config.CVAE_EPOCHS,
                lr=config.CVAE_LR,
                beta=config.CVAE_BETA,
                target_epsilon=config.TARGET_EPSILON,
                target_delta=config.TARGET_DELTA,
                max_grad_norm=config.MAX_GRAD_NORM,
                device=self.device,
            )
            synthetic_embeddings, synthetic_labels = (
                dp_cvae.generate_synthetic_embeddings_from_decoder(
                    result.decoder,
                    real_labels=real_labels,
                    num_classes=config.NUM_CLASSES,
                    latent_dim=config.CVAE_LATENT_DIM,
                    device=self.device,
                )
            )
        else:
            result = dp_cvae.train_local_dp_cvae(
                cvae_train_embeddings,
                real_labels,
                input_dim=self.extractor.embedding_dim,
                num_classes=config.NUM_CLASSES,
                latent_dim=config.CVAE_LATENT_DIM,
                hidden_dim=config.CVAE_HIDDEN_DIM,
                batch_size=config.CVAE_BATCH_SIZE,
                epochs=config.CVAE_EPOCHS,
                lr=config.CVAE_LR,
                beta=config.CVAE_BETA,
                use_dp=config.USE_LOCAL_DP_CVAE,
                target_epsilon=config.TARGET_EPSILON,
                target_delta=config.TARGET_DELTA,
                max_grad_norm=config.MAX_GRAD_NORM,
                device=self.device,
            )
            synthetic_embeddings, synthetic_labels = dp_cvae.generate_synthetic_embeddings(
                result.model,
                real_labels=real_labels,
                num_classes=config.NUM_CLASSES,
                latent_dim=config.CVAE_LATENT_DIM,
                device=self.device,
            )

        if config.NORMALIZE_EMBEDDINGS and emb_mean is not None and emb_std is not None:
            synthetic_embeddings = self._denormalize_embeddings(
                synthetic_embeddings, emb_mean, emb_std
            )

        epsilon = result.epsilon
        cvae_loss = result.final_loss
        metadata = {
            **expected_metadata,
            "epsilon": epsilon,
            "cvae_final_loss": cvae_loss,
            "num_real_examples": len(real_labels),
            "num_synthetic_examples": len(synthetic_labels),
        }
        dp_cvae.save_synthetic_dataset(cache_path, synthetic_embeddings, synthetic_labels, metadata)
        print(
            f"[Cliente {self.partition_id}] Sintéticos salvos em {cache_path}. "
            f"ε_CVAE={epsilon if epsilon is not None else 'sem DP'}"
        )

        synthetic_loader = models.make_embedding_loader(
            synthetic_embeddings,
            synthetic_labels,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
        )
        return synthetic_loader, epsilon, cvae_loss, len(synthetic_labels)

    def _build_or_load_test_embedding_loader(self):
        """Carrega ou extrai os embeddings do conjunto de teste local."""
        return extractor.build_or_load_embedding_loader(
            cache_path=self._test_embedding_cache_path(),
            extractor=self.extractor,
            image_loader=self.test_loader,
            device=self.device,
            metadata=self._test_embedding_metadata(),
            batch_size=config.BATCH_SIZE,
        )

    def fit(self, parameters, config_dict):
        models.set_weights(self.model, parameters)
        self.model.to(self.device)

        # Seleciona automaticamente o loader de acordo com TRAIN_MODE
        if config.TRAIN_MODE == "baseline":
            train_loader, epsilon, cvae_loss, num_samples = self._load_baseline_loader()
        else:
            train_loader, epsilon, cvae_loss, num_samples = self._build_or_load_synthetic_loader()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LR)

        models.train_classifier(
            self.model,
            train_loader,
            optimizer,
            device=self.device,
            epochs=config.LOCAL_EPOCHS,
        )

        metrics = {
            "num_samples": float(num_samples),
        }
        if epsilon is not None:
            metrics["epsilon_cvae"] = float(epsilon)
        if cvae_loss is not None:
            metrics["cvae_loss"] = float(cvae_loss)

        return models.get_weights(self.model), num_samples, metrics

    def evaluate(self, parameters, config_dict):
        models.set_weights(self.model, parameters)
        test_loader = self._build_or_load_test_embedding_loader()
        loss, metrics = models.evaluate_classifier(
            self.model, test_loader, self.device, num_classes=config.NUM_CLASSES
        )
        return loss, len(self.test_loader.dataset), metrics


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    train_loader, test_loader = dataset.get_client_loaders(
        partition_id=partition_id,
        num_partitions=config.NUM_PARTITIONS,
    )

    return FlowerClient(
        train_loader=train_loader,
        test_loader=test_loader,
        partition_id=partition_id,
        context=context,
    ).to_client()


client_app = ClientApp(client_fn=client_fn)