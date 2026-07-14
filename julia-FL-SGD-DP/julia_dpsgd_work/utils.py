import torch

@torch.no_grad()
def calculate_mean_std(loader):
    channel_sum = None
    channel_squared_sum = None
    total_pixels = 0

    for batch in loader:
        images = batch["image"].float()

        if images.dim() == 3:
            images = images.unsqueeze(1)

        if channel_sum is None:
            num_channels = images.size(1)
            channel_sum = torch.zeros(
                num_channels,
                dtype=torch.float64,
            )
            channel_squared_sum = torch.zeros(
                num_channels,
                dtype=torch.float64,
            )

        images = images.to(dtype=torch.float64)

        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (
            images ** 2
        ).sum(dim=(0, 2, 3))

        total_pixels += (
            images.size(0)
            * images.size(2)
            * images.size(3)
        )

    if total_pixels == 0:
        raise ValueError(
            "O loader de pré-treinamento está vazio"
        )

    mean = channel_sum / total_pixels

    variance = (
        channel_squared_sum / total_pixels
        - mean ** 2
    )

    std = torch.sqrt(
        torch.clamp(variance, min=1e-12)
    )

    return (
        mean.to(torch.float32).tolist(),
        std.to(torch.float32).tolist(),
    )