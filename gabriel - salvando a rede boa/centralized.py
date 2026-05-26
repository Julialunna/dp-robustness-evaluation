from datasets import load_dataset


import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import train
from opacus import PrivacyEngine
import parameters_centralized

def load_dataset_centralized(dataset, batch_size):
    pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_data = dataset["train"].with_transform(apply_transforms)
    test_data = dataset["test"].with_transform(apply_transforms)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size)
    
    return trainloader, testloader


def run_centralized_dp(trainloader, testloader, num_classes, epochs:int, lr:float, momentum:float=0.9):
    model = train.Net(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    privacy_engine = PrivacyEngine(secure_mode=False)

    (model, optim, trainloader) = privacy_engine.make_private(
                                                      module=model,
                                                      optimizer=optim,
                                                      data_loader=trainloader,
                                                      noise_multiplier=parameters_centralized.NOISE_MULTIPLIER,
                                                      max_grad_norm=parameters_centralized.MAX_GRAD_NORM,
                                                      )

    model.to(device)

    # Scheduler de Learning Rate

    for e in range(epochs):
        print(f"Training epoch {e} ...")
        # train.train(model, trainloader, optim, device)
        epsilon = train.train(
            model,
            trainloader,
            privacy_engine,
            optim,
            parameters_centralized.TARGET_DELTA,
            device=device,
        )
        print(f"  ε = {epsilon:.2f}")
                

    loss, accuracy = train.test(model, testloader, device)
    print(f"{loss = }")
    print(f"{accuracy = }")





my_dataset = load_dataset("ylecun/mnist")
trainloader, testloader = load_dataset_centralized(my_dataset, batch_size=parameters_centralized.BATCH_SIZE)
run_centralized_dp(trainloader, testloader, 10, epochs=parameters_centralized.EPOCHS, lr=parameters_centralized.LR)