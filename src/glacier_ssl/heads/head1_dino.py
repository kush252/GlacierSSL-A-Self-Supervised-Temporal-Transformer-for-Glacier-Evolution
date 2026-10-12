"""Phase 9 — Head 1: Temporal DINO (head1_archi.md).

Operates on TIMESTAMP embeddings (T,768) from the shared backbone, one set per view.
Student sees all views; teacher sees the global views only; teacher is EMA + centered,
no gradients. Cross-view cross-entropy excludes the matching same-view pair and averages
over valid timestamps and valid view pairs. No explicit negatives (that is the diagram's
version, not this spec).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class DINOHead(nn.Module):
    """Projector (768->2048->2048->bottleneck) + weight-normalized prototype layer."""

    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int,
                 num_prototypes: int, norm_last_layer: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, num_prototypes, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)        # standard DINO: normalize bottleneck
        return self.last_layer(x)              # prototype logits (NOT normalized)


class DINOLoss(nn.Module):
    """Teacher-centered cross-view cross-entropy with EMA center."""

    def __init__(self, num_prototypes: int, student_temp: float, teacher_temp: float,
                 center_momentum: float):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, num_prototypes))

    def forward(self, student_logits: list[torch.Tensor],
                teacher_logits: list[torch.Tensor],
                student_valid: list[torch.Tensor],
                teacher_valid: list[torch.Tensor],
                teacher_is_global: list[int], student_global_index: list[int]):
        """Each logits tensor is (N_t, K) for a view's valid-or-all timestamps stacked as
        (B*T, K); valid masks are (B*T,) bool. ``teacher_is_global`` maps each teacher view
        to its index among all views; ``student_global_index`` the same for students, so we
        can exclude same-view pairs.
        """
        teacher_probs = []
        for tl in teacher_logits:
            p = F.softmax((tl - self.center) / self.teacher_temp, dim=-1)
            teacher_probs.append(p.detach())

        total, n_terms = 0.0, 0
        for ti, (tp, tv) in enumerate(zip(teacher_probs, teacher_valid)):
            t_view = teacher_is_global[ti]
            for si, (sl, sv) in enumerate(zip(student_logits, student_valid)):
                if student_global_index[si] == t_view:
                    continue                    # exclude same-view pair
                valid = tv & sv                 # both views valid at that timestamp
                if valid.sum() == 0:
                    continue
                log_s = F.log_softmax(sl[valid] / self.student_temp, dim=-1)
                loss = -(tp[valid] * log_s).sum(dim=-1).mean()
                total = total + loss
                n_terms += 1
        if n_terms == 0:
            return student_logits[0].sum() * 0.0
        return total / n_terms

    @torch.no_grad()
    def update_center(self, teacher_logits: list[torch.Tensor],
                      teacher_valid: list[torch.Tensor]):
        valid_logits = [tl[v] for tl, v in zip(teacher_logits, teacher_valid)
                        if v.sum() > 0]
        if not valid_logits:
            return
        cat = torch.cat(valid_logits, dim=0)
        if cat.numel() == 0:
            return
        batch_center = cat.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1 - self.center_momentum)


class Head1DINO(nn.Module):
    """Student DINO head + an EMA teacher DINO head (teacher backbone EMA is the trainer's
    responsibility, since the backbone is shared across heads)."""

    def __init__(self, config: Config):
        super().__init__()
        c = config.head1
        dim = int(config.tet.embed_dim)
        args = (dim, int(c.hidden_dim), int(c.bottleneck_dim),
                int(c.num_prototypes), bool(c.norm_last_layer))
        self.student_head = DINOHead(*args)
        # Build teacher independently (weight_norm modules are not deepcopy-safe) and
        # initialize it to the student's weights.
        self.teacher_head = DINOHead(*args)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        self.loss = DINOLoss(int(c.num_prototypes), float(c.student_temp),
                             float(c.teacher_temp), float(c.center_momentum))

    @torch.no_grad()
    def ema_update_head(self, momentum: float):
        for ps, pt in zip(self.student_head.parameters(),
                          self.teacher_head.parameters()):
            pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)
