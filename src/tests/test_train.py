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


def test_resume_restores_state_and_continues(tmp_path):
    """A 12h session cap means long runs MUST resume exactly where they stopped."""
    from glacier_ssl.train.checkpoint import load_checkpoint

    cfg = small_config()
    cfg.set("train.ckpt_every", 0)
    batches = [_synthetic_batch(cfg) for _ in range(2)]

    t1 = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    steps = t1.fit(batches, epochs=2)
    assert steps == 4                                   # 2 epochs x 2 batches

    ckpt = tmp_path / "ckpt_latest.pt"
    assert ckpt.exists(), "rolling checkpoint must be written every epoch"
    assert not (tmp_path / "ckpt_epoch0.pt").exists(), "ckpt_every=0 -> no milestones"

    # a fresh trainer starts different, then matches exactly after loading
    t2 = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    p1 = next(t1.student.vit.parameters()).detach()
    assert not torch.allclose(p1, next(t2.student.vit.parameters()).detach())

    heads = {"head1": t2.head1, "head2": t2.head2, "head3": t2.head3}
    step = load_checkpoint(ckpt, t2.student, t2.teacher, heads, t2.optimizer)
    assert step == 4
    for a, b in zip(t1.student.parameters(), t2.student.parameters()):
        assert torch.allclose(a.detach(), b.detach())
    for a, b in zip(t1.teacher.parameters(), t2.teacher.parameters()):
        assert torch.allclose(a.detach(), b.detach())

    # resuming continues the step counter instead of restarting at 0
    total = t2.fit(batches, epochs=4, start_epoch=step // len(batches), start_step=step)
    assert total == 8


def test_ckpt_every_writes_milestones(tmp_path):
    cfg = small_config()
    cfg.set("train.ckpt_every", 2)
    batches = [_synthetic_batch(cfg)]
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    trainer.fit(batches, epochs=4)
    assert (tmp_path / "ckpt_latest.pt").exists()
    # milestones at the end of epochs 1 and 3 only
    assert (tmp_path / "ckpt_epoch1.pt").exists()
    assert (tmp_path / "ckpt_epoch3.pt").exists()
    assert not (tmp_path / "ckpt_epoch0.pt").exists()
    assert not (tmp_path / "ckpt_epoch2.pt").exists()


def test_teacher_is_ema_only(tmp_path):
    cfg = small_config()
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    for p in trainer.teacher.parameters():
        assert not p.requires_grad
    t0 = next(trainer.teacher.vit.parameters()).clone()
    trainer.train_step(_synthetic_batch(cfg), step=0, total_steps=4)
    t1 = next(trainer.teacher.vit.parameters())
    assert not torch.allclose(t0, t1)               # teacher moved, but only via EMA


def test_lr_warmup_ramps_and_saturates(tmp_path):
    cfg = small_config(train={"warmup_steps": 10})
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    base = float(cfg.train.lr)
    assert trainer._apply_warmup(0) == base * 0.1
    assert trainer._apply_warmup(4) == base * 0.5
    assert trainer._apply_warmup(9) == base
    assert trainer._apply_warmup(500) == base          # clamped, never exceeds base
    assert trainer.optimizer.param_groups[0]["lr"] == base


def test_frozen_prototypes_do_not_move_then_resume(tmp_path):
    """Uniform prototype output is a zero-gradient fixed point, so the layer is held
    still early while the backbone learns something to assign."""
    cfg = small_config(head1={"freeze_last_layer_steps": 2})
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    proto = trainer.head1.student_head.last_layer.weight_v
    backbone = next(trainer.student.vit.parameters())

    before_p, before_b = proto.clone(), backbone.clone()
    trainer.train_step(_synthetic_batch(cfg), step=0, total_steps=8)
    assert torch.allclose(before_p, proto), "prototypes must be frozen at step 0"
    assert not torch.allclose(before_b, backbone), "backbone must still train"

    before_p = proto.clone()
    trainer.train_step(_synthetic_batch(cfg), step=2, total_steps=8)
    assert not torch.allclose(before_p, proto), "prototypes must train once unfrozen"
