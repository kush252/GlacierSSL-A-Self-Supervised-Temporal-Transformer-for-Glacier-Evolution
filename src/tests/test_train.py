"""Phase 12 test: a combined multi-head step runs with finite losses, gradients reach the
shared backbone, and per-loss logging works — using synthetic in-memory batches."""
import torch

from _helpers import small_config
from glacier_ssl.data.sampling import CropGeometry, SpatialAug, RadiometricAug
from glacier_ssl.train.trainer import SSLTrainer


def _synthetic_batch(cfg, B=2, T=3, n_pairs=3, n_intervals=4):
    o = cfg.data.image_size
    base = torch.randn(B, T, 6, o, o)
    tmask = torch.ones(B, T, dtype=torch.bool)
    elapsed = torch.rand(B, T).sort(dim=1).values * 4
    span = elapsed[:, -1] - elapsed[:, 0]
    seasons = [["DJF", "MAM", "JJA"][:T] for _ in range(B)]

    def view(kind):
        return {"img": torch.randn(B, T, 6, o, o),
                "timestamp_mask": tmask.clone(),
                "crop_valid": torch.ones(B, T, dtype=torch.bool),
                "view_type": kind}
    dino_views = ([view("global") for _ in range(cfg.head1.num_global_student)]
                  + [view("local") for _ in range(cfg.head1.num_local_student)])

    pairs = [(0, 0, 1), (1, 1, 2), (0, 0, 2)][:n_pairs]

    intervals = []
    for k in range(n_intervals):
        intervals.append({
            "batch_index": 0, "i": 0, "j": 1, "gap_years": 1.0 + k,
            "season_i": "DJF", "season_j": "JJA",
            "view_a_img": torch.randn(2, 6, o, o), "view_a_valid": torch.ones(2, dtype=torch.bool),
            "view_b_img": torch.randn(2, 6, o, o), "view_b_valid": torch.ones(2, dtype=torch.bool),
        })
    return {"base": base, "timestamp_mask": tmask, "base_valid": tmask.clone(),
            "elapsed_years": elapsed, "span_years": span, "seasons": seasons,
            "glims_id": ["G1", "G2"], "T": [T, T], "dino_views": dino_views,
            "distance_pairs": pairs, "intervals": intervals}


def test_combined_step_finite_and_backbone_grads(tmp_path):
    cfg = small_config()
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    batch = _synthetic_batch(cfg)

    # snapshot a backbone param to confirm it updates
    p0 = next(trainer.student.vit.parameters()).clone()
    logs = trainer.train_step(batch, step=0, total_steps=4)

    for key in ("loss_total", "loss_dino", "loss_distance", "loss_change"):
        assert key in logs and torch.isfinite(torch.tensor(logs[key]))
    assert logs["n_pairs"] == 3 and logs["n_intervals"] == 4
    assert logs["backbone_grad_norm"] > 0           # gradient flows into the backbone
    p1 = next(trainer.student.vit.parameters())
    assert not torch.allclose(p0, p1)               # backbone params updated


def test_teacher_is_ema_only(tmp_path):
    cfg = small_config()
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    for p in trainer.teacher.parameters():
        assert not p.requires_grad
    t0 = next(trainer.teacher.vit.parameters()).clone()
    trainer.train_step(_synthetic_batch(cfg), step=0, total_steps=4)
    t1 = next(trainer.teacher.vit.parameters())
    assert not torch.allclose(t0, t1)               # teacher moved, but only via EMA
