import torch
import torch.nn as nn
import torchvision.models as models


def _build_resnet18_mlp(dim=128):
    """
    ResNet-18 backbone with a 2-layer MLP projection head (for MoCo v2).
    """
    encoder = models.resnet18(weights=None)
    in_dim = encoder.fc.in_features
    encoder.fc = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(inplace=True),
        nn.Linear(in_dim, dim)
    )
    return encoder


class MoCoV2(nn.Module):
    """
    Minimal MoCo v2 implementation with:
    - encoder_q: query encoder
    - encoder_k: momentum encoder
    - queue: dictionary of negative keys
    """

    def __init__(self, dim=128, K=8192, m=0.999, T=0.2):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        # encoders
        self.encoder_q = _build_resnet18_mlp(dim=dim)
        self.encoder_k = _build_resnet18_mlp(dim=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # no grad to key encoder

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the queue with the latest keys.
        """
        batch_size = keys.shape[0]

        K = self.K
        ptr = int(self.queue_ptr)
        assert K % batch_size == 0, "For simplicity, set K divisible by batch_size"

        # replace the keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: query images [N, 3, H, W]
            im_k: key images   [N, 3, H, W]
        Output:
            logits, labels  (for cross-entropy)
        """
        # compute query features
        q = self.encoder_q(im_q)     # [N, dim]
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            # shuffle BN can be added here if using multi-GPU; we skip for simplicity

            k = self.encoder_k(im_k)  # [N, dim]
            k = nn.functional.normalize(k, dim=1)

        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # [N,1]

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])  # [N,K]

        # logits: [N, 1+K]
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positives are first index
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # update queue
        self._dequeue_and_enqueue(k)

        return logits, labels
