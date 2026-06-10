from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import train
from opacus import PrivacyEngine
import parameters_centralized
from VAE_training import VAE
import torch.nn.functional as F

def load_dataset_centralized(dataset, batch_size):
    pytorch_transforms = Compose([ToTensor()])

    
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_data = dataset["train"].with_transform(apply_transforms)
    test_data = dataset["test"].with_transform(apply_transforms)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size)
    
    return trainloader, testloader

def run_centralized_training(trainloader, testloader, num_classes, epochs:int, lr:float, momentum:float=0.9):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train.MLP(  input_size=28 * 28,
    hidden_size=256,
    num_classes=num_classes).to(device)
    
        
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

     
    vae = VAE()
    weights = torch.load("trained_vae.pth", map_location=device)
    vae.load_state_dict(weights)
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        print(f"Training epoch {epoch} ...")

        model.train()
        correct, total_loss = 0, 0
        
        for batch in trainloader:
            labels = batch["label"].to(device)
            batch_size = labels.size(0) 
            y_one_hot = F.one_hot(labels, num_classes=10).float().to(device)
            z_aleatorio = torch.randn(batch_size, 20).to(device)
            
            with torch.no_grad():
                private_images = vae.decode(z_aleatorio, y_one_hot)
            
            optimizer.zero_grad()
            outputs = model(private_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(trainloader):.4f} Acc Treino: {correct/len(trainloader.dataset):.4f}")
        
                

    loss, accuracy = train.test(model, testloader, device)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy*100}%")


if __name__ == "__main__":
    my_dataset = load_dataset("ylecun/mnist")
    trainloader, testloader = load_dataset_centralized(my_dataset, batch_size=parameters_centralized.BATCH_SIZE)
    run_centralized_training(trainloader, testloader, 10, epochs=parameters_centralized.EPOCHS, lr=parameters_centralized.LR)
    
# def run_centralized_training(trainloader, testloader, num_classes, epochs:int, lr:float, momentum:float=0.9):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = train.MLP(  input_size=28 * 28,
#     hidden_size=256,
#     num_classes=num_classes)

#     optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
#     privacy_engine = None

#     if parameters_centralized.USE_DP:
#         privacy_engine = PrivacyEngine(secure_mode=False)

#         model, optim, trainloader = privacy_engine.make_private(
#             module=model,
#             optimizer=optim,
#             data_loader=trainloader,
#             noise_multiplier=parameters_centralized.NOISE_MULTIPLIER,
#             max_grad_norm=parameters_centralized.MAX_GRAD_NORM,
#         )

#     model.to(device)

#     # Scheduler de Learning Rate

#     for e in range(epochs):
#         print(f"Training epoch {e} ...")
#         # train.train(model, trainloader, optim, device)
#         if parameters_centralized.USE_DP:
#             epsilon = train.train(
#                 model,
#                 trainloader,
#                 privacy_engine,
#                 optim,
#                 parameters_centralized.TARGET_DELTA,
#                 device=device,
#             )
#             print(f"  ε = {epsilon:.2f}")
#         else: 
#              train.train(
#                 model,
#                 trainloader,
#                 privacy_engine,
#                 optim,
#                 parameters_centralized.TARGET_DELTA,
#                 device=device,
#             )
                

#     loss, accuracy = train.test(model, testloader, device)
#     print(f"Loss: {loss}")
#     print(f"Accuracy: {accuracy*100}%")
#run_centralized_training(trainloader, testloader, 10, epochs=parameters_centralized.EPOCHS, lr=parameters_centralized.LR)