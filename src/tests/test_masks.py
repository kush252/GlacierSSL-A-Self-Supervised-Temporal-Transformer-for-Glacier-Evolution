"""Phase 13 — mask inertness, variable-length collate, and tiny-batch overfit stability."""
import torch

from _helpers import small_config
from glacier_ssl.backbone.layers import Attention
from glacier_ssl.data.dataset import collate_windows
from glacier_ssl.train.trainer import SSLTrainer
from tests.test_train import _synthetic_batch   # reuse synthetic batch builder


# ----------------------------------------------------------- attention key-padding
def test_attention_key_padding_inert():
    """Adding masked (invalid) key positions must not change outputs at valid queries."""
    torch.manual_seed(0)
    attn = Attention(dim=32, num_heads=4).eval()
    x = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        out = attn(x, key_padding_mask=mask)
        x2 = torch.cat([x, torch.randn(1, 3, 32)], dim=1)       # append junk keys
        mask2 = torch.cat([mask, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
        out2 = attn(x2, key_padding_mask=mask2)
    assert torch.allclose(out[0], out2[0, :5], atol=1e-6)


# ----------------------------------------------------------- collate variable length
def test_collate_pads_variable_T_and_masks():
    cfg = small_config()
    o = cfg.data.image_size

    def sample(T):
        s = {"glims_id": "G", "split": "train", "T": T,
             "dates": ["2016-01-01"] * T, "seasons": ["DJF"] * T,
             "elapsed_years": torch.arange(T, dtype=torch.float32),
             "span_years": float(T - 1), "observation_ids": list(range(T)),
             "base_img": torch.randn(T, 6, o, o),
             "base_valid": torch.ones(T, dtype=torch.bool),
             "sample_spec": None}
        return s

    batch = [sample(2), sample(5)]
    out = collate_windows(batch)
    assert out["base"].shape == (2, 5, 6, o, o)          # padded to max T=5
    assert out["timestamp_mask"][0].tolist() == [True, True, False, False, False]
    assert out["timestamp_mask"][1].all()
    # padded region of the short sample is zero
    assert torch.count_nonzero(out["base"][0, 2:]) == 0


def test_collate_flattens_pairs_and_intervals_with_batch_index():
    cfg = small_config()
    o = cfg.data.image_size

    def sample(T, gid):
        return {"glims_id": gid, "split": "train", "T": T, "dates": ["2016-01-01"] * T,
                "seasons": ["DJF"] * T, "elapsed_years": torch.arange(T, dtype=torch.float32),
                "span_years": float(T - 1), "observation_ids": list(range(T)),
                "base_img": torch.randn(T, 6, o, o), "base_valid": torch.ones(T, dtype=torch.bool),
                "sample_spec": None, "distance_pairs": [(0, 1)],
                "intervals": [{"i": 0, "j": 1, "gap_years": 1.0, "season_i": "DJF",
                               "season_j": "JJA",
                               "view_a_img": torch.randn(2, 6, o, o),
                               "view_a_valid": torch.ones(2, dtype=torch.bool),
                               "view_b_img": torch.randn(2, 6, o, o),
                               "view_b_valid": torch.ones(2, dtype=torch.bool)}]}

    out = collate_windows([sample(3, "A"), sample(4, "B")])
    assert [p[0] for p in out["distance_pairs"]] == [0, 1]       # batch indices preserved
    assert [iv["batch_index"] for iv in out["intervals"]] == [0, 1]


# ----------------------------------------------------------- tiny-batch overfit
def test_combined_overfits_tiny_batch(tmp_path):
    """Stability + learning on a fixed batch. The cleanly-overfittable heads (distance,
    change) must decrease; the total must stay finite and bounded (no instability). DINO
    is not asserted to decrease monotonically — its spec asks for collapse *monitoring*,
    and per-head DINO/change overfit is covered in test_heads."""
    cfg = small_config(train={"lr": 1e-3})
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    batch = _synthetic_batch(cfg)
    steps = 30
    dist, change, total = [], [], []
    for step in range(steps):
        logs = trainer.train_step(batch, step=step, total_steps=steps)
        total.append(logs["loss_total"])
        dist.append(logs["loss_distance"])
        change.append(logs["loss_change"])
        assert torch.isfinite(torch.tensor(logs["loss_total"]))   # no NaN/Inf
        assert logs["loss_total"] < 1e3                           # no explosion
    assert dist[-1] < dist[0]                                     # distance head overfits
    assert change[-1] < change[0]                                 # change head overfits
