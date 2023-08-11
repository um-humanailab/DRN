from syslog import LOG_AUTHPRIV
import torch
from torch.nn import nn

class JSD_Loss(nn.Module):
    def __init__(self) -> None:
        super(JSD_Loss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.view(-1))
        m = 

