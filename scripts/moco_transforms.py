import torchvision.transforms as T

class TwoCropsTransform:
    """
    Take one image and return two differently augmented views.
    This is the standard MoCo / SimCLR-style transform wrapper.
    """
    def __init__(self):
        color_jitter = T.ColorJitter(0.4, 0.4, 0.2, 0.1)

        base_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
