import warnings
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import logging
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

warnings.filterwarnings("ignore", category=UserWarning)

import train
import parameters_federated


# NUM_PARTITIONS = 100

# TARGET_DELTA = 1e-5 # probabilidade de falha da garantia de privacidade.
# MAX_GRAD_NORM = 1.0 #Limiar de clipping C — cada gradiente por amostra é clipado para ter norma L2 ≤ C
# NOISE_MULTIPLIER = 1.1  # ruído gaussiano
# NUM_CLASSES = 10

class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_loader,
        test_loader,
        target_delta,
        noise_multiplier,
        max_grad_norm,
    ) -> None:
        super().__init__()
        self.model = train.MLP(parameters_federated.NUM_CLASSES)

        # errors = ModuleValidator.validate(self.model, strict=False)

        # if errors:
        #     print(f"Encontrados {len(errors)} problemas de compatibilidade")
        #     self.model = ModuleValidator.fix(self.model)
        #     print("Modelo corrigido automaticamente pelo Opacus")
        # else:
        #     print("Modelo compatível com Opacus")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        model = self.model
        model.to(self.device)
        train.set_weights(model, parameters)

        optimizer = torch.optim.Adam(model.parameters(), lr=parameters_federated.LR)

        privacy_engine = PrivacyEngine(secure_mode=False)
        (model, optimizer, self.train_loader) = privacy_engine.make_private(
                                                      module=model,
                                                      optimizer=optimizer,
                                                      data_loader=self.train_loader,
                                                      noise_multiplier=self.noise_multiplier,
                                                      max_grad_norm=self.max_grad_norm,
                                                      grad_sample_mode="ew"
                                                      )

        epsilon = train.train(
            model,
            self.train_loader,
            privacy_engine,
            optimizer,
            self.target_delta,
            device=self.device,
            epochs=parameters_federated.EPOCHS,
        )

        if epsilon is not None:
            print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
        else:
            print("Epsilon value not available.")

        return (train.get_weights(model), len(self.train_loader.dataset), {})

    def evaluate(self, parameters, config):
        train.set_weights(self.model, parameters)
        loss, accuracy = train.test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    # noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5

    train_loader, test_loader = train.load_data(
        partition_id=partition_id, num_partitions=parameters_federated.NUM_PARTITIONS
    )

    return FlowerClient(
        train_loader,
        test_loader,
        parameters_federated.TARGET_DELTA,
        parameters_federated.NOISE_MULTIPLIER,
        parameters_federated.MAX_GRAD_NORM,
    ).to_client()


client_app = ClientApp(client_fn=client_fn)