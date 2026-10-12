"""Phase 12 — Combined SSL training loop (joint three-head training).

Wires the three student-side objectives into the shared backbone and the EMA teacher:

* Head 1 (DINO): teacher backbone+head process the GLOBAL views (no grad); student
  backbone+head process ALL views; cross-view CE over valid timestamps.
* Head 2 (distance): student backbone on the base view -> timestamp_out -> sampled pairs.
* Head 3 (change): student backbone on each interval's two augmented views -> regional_out.

L_total = λ_dino·L_DINO + λ_distance·L_distance + λ_change·L_change (coefficients in config).
Each loss, gradient norm to the backbone, and valid-sample counts are logged separately
(a coefficient is not influence). Teacher = EMA only; all student paths backprop into the
shared ViT+HPE+TET.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from ..backbone.backbone import GlacierBackbone
from ..backbone.tet import seasons_to_ids
from ..config import Config
from ..heads.head1_dino import Head1DINO
from ..heads.head2_distance import Head2Distance
from ..heads.head3_change import Head3Change
from ..utils.logging_utils import get_logger, MetricLogger
from .checkpoint import save_checkpoint


def _cosine(step, total, start, end):
    if total <= 1:
        return end
    return end - (end - start) * (1 + math.cos(math.pi * step / total)) / 2


class SSLTrainer:
    def __init__(self, config: Config, output_dir: str | Path, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("trainer")
        self.metrics = MetricLogger(self.out / "metrics.jsonl")

        self.student = GlacierBackbone(config).to(self.device)
        self.teacher = GlacierBackbone(config).to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.head1 = Head1DINO(config).to(self.device)
        self.head2 = Head2Distance(config).to(self.device)
        self.head3 = Head3Change(config).to(self.device)

        params = (list(self.student.parameters())
                  + list(self.head1.student_head.parameters())
                  + list(self.head2.parameters())
                  + list(self.head3.parameters()))
        self.optimizer = torch.optim.AdamW(
            params, lr=float(config.train.lr), weight_decay=float(config.train.weight_decay))
        self.grad_clip = float(config.train.grad_clip)
        self.lambda_dino = float(config.loss.lambda_dino)
        self.lambda_distance = float(config.loss.lambda_distance)
        self.lambda_change = float(config.loss.lambda_change)
        self.n_global = int(config.head1.num_global_teacher)

    # -- EMA ---------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, momentum: float):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)
        for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
            bt.copy_(bs)
        self.head1.ema_update_head(momentum)

    # -- per-head loss computation ----------------------------------------------
    def _dino_loss(self, batch):
        views = batch["dino_views"]
        elapsed = batch["elapsed_years"].to(self.device)
        span = batch["span_years"].to(self.device)
        seasons = batch["seasons"]

        global_view_indices = [v for v, vw in enumerate(views)
                               if vw["view_type"] == "global"]
        # student logits for every view
        student_logits, student_valid, student_global_index = [], [], []
        for v, vw in enumerate(views):
            img = vw["img"].to(self.device)
            tmask = vw["timestamp_mask"].to(self.device)
            out = self.student(img, tmask, elapsed, span, seasons)
            ts = out["timestamp_out"]                           # (B,T,D)
            B, T, D = ts.shape
            logits = self.head1.student_head(ts.reshape(B * T, D))
            valid = (tmask & vw["crop_valid"].to(self.device)).reshape(B * T)
            student_logits.append(logits)
            student_valid.append(valid)
            student_global_index.append(v)

        # teacher logits for global views only (no grad)
        teacher_logits, teacher_valid, teacher_is_global = [], [], []
        with torch.no_grad():
            for v in global_view_indices:
                vw = views[v]
                img = vw["img"].to(self.device)
                tmask = vw["timestamp_mask"].to(self.device)
                out = self.teacher(img, tmask, elapsed, span, seasons)
                ts = out["timestamp_out"]
                B, T, D = ts.shape
                logits = self.head1.teacher_head(ts.reshape(B * T, D))
                valid = (tmask & vw["crop_valid"].to(self.device)).reshape(B * T)
                teacher_logits.append(logits)
                teacher_valid.append(valid)
                teacher_is_global.append(v)

        loss = self.head1.loss(student_logits, teacher_logits, student_valid,
                               teacher_valid, teacher_is_global, student_global_index)
        self.head1.loss.update_center(teacher_logits, teacher_valid)
        n_valid = int(sum(v.sum() for v in student_valid))
        return loss, {"n_dino_valid": n_valid}

    def _distance_loss(self, batch, base_out):
        pairs = batch.get("distance_pairs", [])
        if not pairs:
            z = base_out["timestamp_out"]
            return z.sum() * 0.0, {"n_pairs": 0}
        ts = base_out["timestamp_out"]                          # (B,T,D)
        elapsed = batch["elapsed_years"]                        # (B,T) dates-derived
        zi, zj, targets = [], [], []
        for (b, i, j) in pairs:
            zi.append(ts[b, i]); zj.append(ts[b, j])
            targets.append(abs(float(elapsed[b, j] - elapsed[b, i])))
        zi = torch.stack(zi); zj = torch.stack(zj)
        target = torch.tensor(targets, device=self.device, dtype=torch.float32)
        pred = self.head2(zi, zj)
        loss, m = self.head2.compute_loss(pred, target)
        m["n_pairs"] = len(pairs)
        return loss, m

    def _change_loss(self, batch):
        intervals = batch.get("intervals", [])
        # keep only intervals whose both endpoints are valid in both views
        usable = [iv for iv in intervals
                  if bool(iv["view_a_valid"].all()) and bool(iv["view_b_valid"].all())]
        if len(usable) < 2:                                     # InfoNCE needs >=2 negatives
            dummy = next(self.head3.parameters())
            return dummy.sum() * 0.0, {"n_intervals": len(usable)}

        def run_view(key):
            imgs = torch.stack([iv[f"{key}_img"] for iv in usable]).to(self.device)  # (N,2,6,o,o)
            N = imgs.shape[0]
            tmask = torch.ones(N, 2, dtype=torch.bool, device=self.device)
            gaps = torch.tensor([iv["gap_years"] for iv in usable], device=self.device)
            elapsed = torch.stack([torch.zeros(N, device=self.device), gaps], dim=1)  # (N,2)
            span = gaps
            seasons = [[iv["season_i"], iv["season_j"]] for iv in usable]
            out = self.student(imgs, tmask, elapsed, span, seasons)
            reg = out["regional_out"]                           # (N,2,64,D)
            return reg[:, 0], reg[:, 1]

        r_i_a, r_j_a = run_view("view_a")
        r_i_b, r_j_b = run_view("view_b")
        loss, m = self.head3(r_i_a, r_j_a, r_i_b, r_j_b)
        m["n_intervals"] = len(usable)
        return loss, m

    # -- one step ----------------------------------------------------------------
    def train_step(self, batch, step, total_steps):
        self.student.train(); self.head1.train(); self.head2.train(); self.head3.train()
        self.optimizer.zero_grad()

        # base-view backbone pass (shared by Head 2); kept with grad.
        base_out = self.student(
            batch["base"].to(self.device), batch["timestamp_mask"].to(self.device),
            batch["elapsed_years"].to(self.device), batch["span_years"].to(self.device),
            batch["seasons"])

        logs = {}
        l_dino, m1 = self._dino_loss(batch); logs.update(m1)
        l_dist, m2 = self._distance_loss(batch, base_out); logs.update(m2)
        l_change, m3 = self._change_loss(batch); logs.update(m3)

        total = (self.lambda_dino * l_dino + self.lambda_distance * l_dist
                 + self.lambda_change * l_change)
        total.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        self.optimizer.step()

        momentum = _cosine(step, total_steps, float(self.cfg.head1.ema_momentum_start),
                           float(self.cfg.head1.ema_momentum_end))
        self._ema_update(momentum)

        logs.update({
            "loss_total": float(total.detach()), "loss_dino": float(l_dino.detach()),
            "loss_distance": float(l_dist.detach()), "loss_change": float(l_change.detach()),
            "backbone_grad_norm": float(grad_norm), "ema_momentum": momentum,
        })
        self.metrics.log(step, **logs)
        return logs

    # -- epoch loop --------------------------------------------------------------
    def fit(self, loader, epochs: int):
        total_steps = max(1, epochs * len(loader))
        step = 0
        for epoch in range(epochs):
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
            for batch in loader:
                logs = self.train_step(batch, step, total_steps)
                if step % int(self.cfg.train.log_every) == 0:
                    self.log.info(
                        "step %d | total %.4f | dino %.4f dist %.4f change %.4f | "
                        "gnorm %.3f | pairs %d intervals %d",
                        step, logs["loss_total"], logs["loss_dino"],
                        logs["loss_distance"], logs["loss_change"],
                        logs["backbone_grad_norm"], logs.get("n_pairs", 0),
                        logs.get("n_intervals", 0))
                step += 1
            save_checkpoint(self.out / f"ckpt_epoch{epoch}.pt", step, self.student,
                            self.teacher, {"head1": self.head1, "head2": self.head2,
                                           "head3": self.head3},
                            self.optimizer, self.cfg)
        self.metrics.close()
        return step
