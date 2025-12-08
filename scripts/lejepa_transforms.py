from typing import List
import torchvision.transforms as T


class LeJEPAMultiCrop:
    """
    LeJEPA-style multi-crop transform.

    Returns:
        List[Tensor] containing:
            - num_global_crops crops of size global_size
            - num_local_crops crops of size local_size
    """

    def __init__(
        self,
        num_global_crops: int = 2,
        num_local_crops: int = 4,
        global_size: int = 224,
        local_size: int = 98,
    ):
        if num_global_crops < 1:
            raise ValueError("num_global_crops must be >= 1")
        if num_local_crops < 0:
            raise ValueError("num_local_crops must be >= 0")

        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

        color_jitter = T.ColorJitter(0.4, 0.4, 0.2, 0.1)

        common_augs = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]

        self.global_transform = T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.3, 1.0)),
            *common_augs,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.05, 0.3)),
            *common_augs,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img) -> List:
        crops = [
            self.global_transform(img)
            for _ in range(self.num_global_crops)
        ]
        crops += [
            self.local_transform(img)
            for _ in range(self.num_local_crops)
        ]
        return crops
