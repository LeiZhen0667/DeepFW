import torch
import torch.nn as nn
import torch.nn.functional as F

class HCTCLoss(nn.Module):
    def __init__(self, margin, lambda_reg, num_classes, encoding_dim, device):
        super(HCTCLoss, self).__init__()
        self.margin = margin
        self.lambda_reg = lambda_reg
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(num_classes, encoding_dim).to(device))

    def forward(self, anchors, negatives, labels):
        batch_size = anchors.size(0)
        batch_centers = torch.zeros_like(anchors)
        for i in range(self.num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                batch_centers[mask] = self.centers[i]

        normalized_anchors = F.normalize(anchors, p=2, dim=1)
        normalized_negatives = F.normalize(negatives, p=2, dim=1)
        normalized_centers = F.normalize(batch_centers, p=2, dim=1)

        similarity_neg = torch.matmul(normalized_anchors, normalized_negatives.t()).max(dim=1)[0]
        similarity_center = (normalized_anchors * normalized_centers).sum(dim=1)

        hctc_loss = F.relu(similarity_neg - similarity_center + self.margin).mean()

        reg_loss = torch.norm(self.centers[labels] - batch_centers, dim=1).sum() * self.lambda_reg

        return hctc_loss + reg_loss
