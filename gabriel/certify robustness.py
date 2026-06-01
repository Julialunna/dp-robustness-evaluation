
#Esse arquivo aqui é pra, após o treinamento, certificar a qualidade da robustez do nosso modelo!


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from flwr_datasets import FederatedDataset
import numpy as np
import train
import parameters_federated

device = "cuda" if torch.cuda.is_available() else "cpu"
NOISE = [0.0, 0.01, 0.05, 0.1, 0.25]

class TupleDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, i):
        return self.transform(self.ds[i]["image"]), self.ds[i]["label"]

def get_test_dataset():
    fds = FederatedDataset(dataset="ylecun/mnist", partitioners={})
    split = fds.load_split("test")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((parameters_federated.MEAN,), (parameters_federated.STD,))
    ])
    return TupleDataset(split, transform)


# considerando que o modelo foi salvo assim: 
# np.save(save_path, params_array, allow_pickle=True)
def load_model(num_classes, path="./modelos/modelo_final_FL_DP.npy", device="cpu"):
    model = train.Net(num_classes)
    params = list(np.load(path, allow_pickle=True))
    state_dict = model.state_dict()
    for i, key in enumerate(state_dict.keys()):
        state_dict[key] = torch.from_numpy(params[i]).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def noise_robustness(model, dataset, noise_levels=NOISE, batch_size=128, device="cuda"):
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = {}
    
    for r in noise_levels:
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if r > 0:
                x = x + torch.randn_like(x) * r
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        results[r] = correct / total
        
    return results

if __name__ == "__main__":
    model = load_model(parameters_federated.NUM_CLASSES, device=device)
    dataset = get_test_dataset()
    summary = noise_robustness(model, dataset, device=device)
    
    print("\nResultados Empíricos de Robustez:")
    for r, acc in summary.items():
        print(f"Ruído (sigma): {r} | Acurácia: {acc:.4f}")


# Pra testar isso na prática e comparar, basta rodar duas vezes
# uma com modelo normal cru e outra com nosso modelo, tipo assim:


# import matplotlib.pyplot as plt

# model_base = load_model(10, "./modelos/modelo_centralizado_cru.npy", device)
# model_dp = load_model(10, "./modelos/modelo_final_FL_DP.npy", device)

# res_base = noise_robustness(model_base, dataset, device=device)
# res_dp = noise_robustness(model_dp, dataset, device=device)

# x = list(res_base.keys())
# plt.plot(x, list(res_base.values()), label="Centralizado (Cru)", marker="o")
# plt.plot(x, list(res_dp.values()), label="CVAE + FL + DP", marker="s")
# plt.xlabel("Sigma (Ruído)")
# plt.ylabel("Acurácia")
# plt.legend()
# plt.grid()
# plt.show()


