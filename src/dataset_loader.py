from torch.utils.data import Subset
from torchvision import transforms
from medmnist import INFO
import medmnist


def get_dataset(flag, size, split, root, max_images=None):
    if flag not in INFO:
        raise ValueError(f"Unknown dataset flag: {flag}")

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])

    dataset = DataClass(
        split=split,
        root=str(root),
        download=True,
        size=size,
        transform=transforms.ToTensor(),
    )

    if max_images is not None:
        max_images = min(max_images, len(dataset))
        dataset = Subset(dataset, range(max_images))

    return dataset, info