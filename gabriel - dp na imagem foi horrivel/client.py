import warnings
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

warnings.filterwarnings("ignore", category=UserWarning)

import train
import parameters_federated
import copy

import pickle
from flwr.common import ConfigRecord


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
        partition_id,
        context,
    ) -> None:
        super().__init__()
        self.model = train.Net(num_classes=parameters_federated.NUM_CLASSES)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.partition_id = int(partition_id)
        self.context = context
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _accountant_key(self):
        return f"accountant_state_client_{self.partition_id}"

    def fit(self, parameters, config):
        model = self.model
        model.to(self.device)
        train.set_weights(model, parameters)
        train_loader = self.train_loader
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters_federated.LR)

        train.train(
            model,
            train_loader,
            optimizer,
            device=self.device,
            epochs=parameters_federated.EPOCHS,
        )        
        
        return (train.get_weights(model), len(self.train_loader.dataset), {})

    def evaluate(self, parameters, config):
        train.set_weights(self.model, parameters)
        loss, accuracy = train.test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]

    train_loader, test_loader = train.load_data(
        partition_id=partition_id, num_partitions=parameters_federated.NUM_PARTITIONS
    )

    return FlowerClient(
        train_loader,
        test_loader,
        parameters_federated.TARGET_DELTA,
        parameters_federated.NOISE_MULTIPLIER,
        parameters_federated.MAX_GRAD_NORM,
        partition_id,
        context,
    ).to_client()


# client_app = ClientApp(client_fn=client_fn)