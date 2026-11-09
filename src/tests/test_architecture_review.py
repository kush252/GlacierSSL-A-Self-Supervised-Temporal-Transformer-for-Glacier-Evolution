"""Tests for the architecture-review changes (src/DOCS_phase0/architecture_review.md).

1. Temporal firewall: a time-blind TET pass carries NO explicit temporal metadata, and the
   trainer routes Heads 2 and 3 through it while DINO keeps the time-aware context.
2. Sparse windows: a window can span the whole record at a fixed number of timestamps.
3. Head 2 never pairs invalid (cloud/nodata) timestamps.
4. Pretrained ViT initialisation: real-world checkpoint layouts map onto the model, and
   pretrained weights get their own (lower) learning rate through warmup.
5. Sentinel-2 baseline harmonisation: the +1000 DN offset of processing baseline >= 04.00
   is removed per scene, after nodata detection, and never for older configs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _helpers import small_config
from glacier_ssl.backbone.tet import TemporalEvolutionTransformer
from glacier_ssl.backbone.vit import MultispectralViT, load_pretrained
from glacier_ssl.data.manifest import GlacierSequence, Observation
from glacier_ssl.data.preprocessing import Preprocessor
from glacier_ssl.data.sampling import WindowSampler
from glacier_ssl.train.trainer import SSLTrainer
from test_train import _synthetic_batch


# --------------------------------------------------------------------------- 1. firewall
def _tet_inputs(cfg, B=2, T=5, R=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    regional = torch.randn(B, T, R, cfg.tet.embed_dim, generator=g)
    tmask = torch.ones(B, T, dtype=torch.bool)
    return regional, tmask


def test_time_blind_output_ignores_all_temporal_metadata():
    cfg = small_config()
    tet = TemporalEvolutionTransformer(cfg).eval()
    regional, tmask = _tet_inputs(cfg)
    meta_a = (torch.tensor([[0, 1, 2, 3, 0]] * 2), torch.tensor([[0., .25, .5, .75, 1.]] * 2),
              torch.tensor([1.0, 1.0]))
    meta_b = (torch.tensor([[2, 2, 0, 1, 3]] * 2), torch.tensor([[0., 3., 4., 6., 8.]] * 2),
              torch.tensor([8.0, 8.0]))
    with torch.no_grad():
        aware_a = tet(regional, *meta_a, tmask)
        aware_b = tet(regional, *meta_b, tmask)
        blind_a = tet(regional, *meta_a, tmask, time_blind=True)
        blind_b = tet(regional, *meta_b, tmask, time_blind=True)
    # time-aware: dates/seasons change the representation (they are meant to)
    assert not torch.allclose(aware_a["timestamp_out"], aware_b["timestamp_out"])
    # time-blind: nothing about dates, seasons or span can reach any output
    for key in ("regional_out", "timestamp_out", "sequence_out"):
        assert torch.equal(blind_a[key], blind_b[key]), key


def test_time_blind_carries_no_order_information():
    """Without the encodings, temporal attention sees an unordered set: permuting the
    timestamps permutes the outputs and nothing else. So a firewalled head cannot read
    the gap from position either -- only from appearance."""
    cfg = small_config()
    tet = TemporalEvolutionTransformer(cfg).eval()
    regional, tmask = _tet_inputs(cfg, B=1, T=5)
    perm = torch.tensor([3, 0, 4, 1, 2])
    z = torch.zeros(1, 5, dtype=torch.long)
    with torch.no_grad():
        out = tet(regional, z, z.float(), torch.zeros(1), tmask, time_blind=True)
        outp = tet(regional[:, perm], z, z.float(), torch.zeros(1), tmask, time_blind=True)
    assert torch.allclose(out["timestamp_out"][:, perm], outp["timestamp_out"], atol=1e-5)
    assert torch.allclose(out["sequence_out"], outp["sequence_out"], atol=1e-5)


def _record_time_blind(monkeypatch, trainer):
    calls = []
    orig = trainer.student.tet.forward.__func__

    def spy(self, *a, **kw):
        calls.append(bool(kw.get("time_blind", False)))
        return orig(self, *a, **kw)
    monkeypatch.setattr(TemporalEvolutionTransformer, "forward", spy)
    return calls


def test_trainer_routes_heads_through_the_firewall(tmp_path, monkeypatch):
    cfg = small_config()                       # default.yaml: h1 aware, h2 + h3 blind
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    calls = _record_time_blind(monkeypatch, trainer)
    trainer.train_step(_synthetic_batch(cfg), step=0, total_steps=4)
    n_dino = (int(cfg.head1.num_global_student) + int(cfg.head1.num_local_student)
              + int(cfg.head1.num_global_teacher))
    # order in train_step: base (Head 2), DINO student + teacher views, Head 3 views a, b
    assert calls[0] is True, "Head 2 base pass must be time-blind"
    assert calls[1:1 + n_dino] == [False] * n_dino, "DINO keeps the temporal context"
    assert calls[1 + n_dino:] == [True, True], "Head 3 interval views must be time-blind"


def test_firewall_off_restores_original_routing(tmp_path, monkeypatch):
    cfg = small_config(head2={"time_blind": False}, head3={"time_blind": False})
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    calls = _record_time_blind(monkeypatch, trainer)
    trainer.train_step(_synthetic_batch(cfg), step=0, total_steps=4)
    assert not any(calls)


# --------------------------------------------------------------------------- 2. windows
def _seq(n=36):
    obs = []
    for k in range(n):
        y = 2016 + k // 4
        obs.append(Observation(glims_id="G", observation_id=k, year=y,
                               season=["DJF", "MAM", "JJA", "SON"][k % 4],
                               acquisition_date=f"{y}-{(k % 4) * 3 + 1:02d}-15",
                               band_paths={}, scl_path=None, cloud_pct=0.0))
    return GlacierSequence("G", "r", "train", 1.0, (0.0, 0.0), obs)


def _window_cfg(sparse_prob):
    return small_config(sampling={"window": {
        "min_length": 8, "max_length": 8, "short_max": 4, "medium_max": 8,
        "length_bucket_freq": {"short": 0.34, "medium": 0.33, "long": 0.33},
        "sparse_prob": sparse_prob}})


def test_sparse_windows_span_years_at_fixed_length():
    seq = _seq(36)
    contiguous = WindowSampler(_window_cfg(0.0))
    sparse = WindowSampler(_window_cfg(1.0))
    spans_c, spans_s = [], []
    for seed in range(30):
        c = contiguous.sample(seq, seed)
        s = sparse.sample(seq, seed)
        assert len(c.obs_indices) == len(s.obs_indices) == 8          # same memory budget
        assert c.obs_indices == list(range(c.obs_indices[0], c.obs_indices[0] + 8))
        assert s.obs_indices == sorted(set(s.obs_indices))            # chronological, unique
        assert s.dates == sorted(s.dates)
        spans_c.append(c.span_years)
        spans_s.append(s.span_years)
    assert max(spans_c) < 2.0                    # 8 seasonal steps cannot exceed ~1.75 y
    assert np.mean(spans_s) > 5.0                # sparse windows reach multi-year gaps


def test_sparse_prob_zero_reproduces_original_windows():
    """Turning the option off must not perturb the RNG stream of existing runs."""
    seq = _seq(36)
    cfg = _window_cfg(0.0)
    cfg_absent = small_config(sampling={"window": {
        "min_length": 8, "max_length": 8, "short_max": 4, "medium_max": 8,
        "length_bucket_freq": {"short": 0.34, "medium": 0.33, "long": 0.33}}})
    cfg_absent._data["sampling"]["window"].pop("sparse_prob", None)
    a, b = WindowSampler(cfg), WindowSampler(cfg_absent)
    for seed in range(10):
        assert a.sample(seq, seed).obs_indices == b.sample(seq, seed).obs_indices


# --------------------------------------------------------------------------- 3. validity
def test_head2_never_pairs_invalid_timestamps(tmp_path):
    cfg = small_config()
    trainer = SSLTrainer(cfg, output_dir=tmp_path, device="cpu")
    batch = _synthetic_batch(cfg)                # pairs (0,0,1) (1,1,2) (0,0,2)
    batch["base_valid"][0, 0] = False            # glacier 0, timestamp 0 is cloud
    trainer.student.eval()
    with torch.no_grad():
        out = trainer.student(batch["base"], batch["timestamp_mask"],
                              batch["elapsed_years"], batch["span_years"], batch["seasons"])
    _, m = trainer._distance_loss(batch, out)
    assert m["n_pairs"] == 1                     # only (1,1,2) survives
    batch["base_valid"][:] = False
    loss, m = trainer._distance_loss(batch, out)
    assert m["n_pairs"] == 0 and torch.isfinite(loss)


# --------------------------------------------------------------------------- 4. pretrained
def _prithvi_like_state(vit: MultispectralViT, grid: int = 4, frames: int = 3) -> dict:
    """A checkpoint laid out the way Prithvi / MAE encoders are: 'encoder.' prefix, a 3-D
    tubelet patch kernel, a cls position plus a multi-frame positional table on a smaller
    grid, and decoder weights that must be ignored."""
    g = torch.Generator().manual_seed(0)
    D, C, p = vit.embed_dim, vit.in_chans, vit.patch_size
    state = {"encoder.patch_embed.proj.weight": torch.randn(D, C, 1, p, p, generator=g),
             "encoder.patch_embed.proj.bias": torch.randn(D, generator=g),
             "encoder.cls_token": torch.randn(1, 1, D, generator=g),
             "encoder.pos_embed": torch.randn(1, 1 + frames * grid * grid, D, generator=g),
             "decoder.decoder_embed.weight": torch.randn(16, D, generator=g)}
    for k, v in vit.state_dict().items():
        if k.startswith("blocks.") or k.startswith("norm."):
            state["encoder." + k] = torch.randn(v.shape, generator=g)
    return state


def test_pretrained_loader_adapts_real_checkpoint_layouts(tmp_path):
    cfg = small_config()
    vit = MultispectralViT(cfg)
    state = _prithvi_like_state(vit)
    path = tmp_path / "prithvi_like.pt"
    torch.save(state, path)
    report = load_pretrained(vit, str(path), cfg)

    sd = vit.state_dict()
    assert report["missing_after_load"] == []               # every ViT tensor initialised
    assert torch.equal(sd["patch_embed.weight"],
                       state["encoder.patch_embed.proj.weight"].squeeze(2))
    assert torch.equal(sd["blocks.0.attn.qkv.weight"], state["encoder.blocks.0.attn.qkv.weight"])
    assert sd["pos_embed"].shape == (1, vit.grid * vit.grid, vit.embed_dim)
    assert any("cls position dropped" in n for n in report["notes"])
    assert any("frames averaged" in n for n in report["notes"])
    assert any("resampled 4x4" in n for n in report["notes"])
    assert not any("decoder" in k for k in report["loaded"])


def test_pretrained_band_mismatch_keeps_fresh_patch_embed(tmp_path):
    cfg = small_config()
    vit = MultispectralViT(cfg)
    fresh = vit.patch_embed.weight.detach().clone()
    fresh_bias = vit.patch_embed.bias.detach().clone()
    state = _prithvi_like_state(vit)
    D, p = vit.embed_dim, vit.patch_size
    state["encoder.patch_embed.proj.weight"] = torch.randn(D, 13, 1, p, p)   # 13 bands
    torch.save(state, tmp_path / "s2_13band.pt")
    report = load_pretrained(vit, str(tmp_path / "s2_13band.pt"), cfg)
    assert torch.equal(vit.patch_embed.weight, fresh)
    assert torch.equal(vit.patch_embed.bias, fresh_bias)     # no orphan bias either
    assert "pretrained_band_indices" in report["reasons"]["patch_embed.weight"]
    # ...and an explicit band mapping makes it load
    cfg2 = small_config(vit={"pretrained_band_indices": [1, 2, 3, 7, 11, 12]})
    vit2 = MultispectralViT(cfg2)
    load_pretrained(vit2, str(tmp_path / "s2_13band.pt"), cfg2)
    assert torch.equal(vit2.patch_embed.weight,
                       state["encoder.patch_embed.proj.weight"].squeeze(2)[:, [1, 2, 3, 7, 11, 12]])


def test_trainer_uses_pretrained_vit_with_lower_lr(tmp_path):
    cfg0 = small_config()
    path = tmp_path / "pre.pt"
    torch.save(_prithvi_like_state(MultispectralViT(cfg0)), path)
    cfg = small_config(vit={"pretrained_ckpt": str(path), "pretrained_lr_scale": 0.1},
                       train={"warmup_steps": 10})
    trainer = SSLTrainer(cfg, output_dir=tmp_path / "run", device="cpu")
    loaded = torch.load(path)
    assert torch.equal(trainer.student.vit.blocks[0].attn.qkv.weight,
                       loaded["encoder.blocks.0.attn.qkv.weight"])
    # teacher starts from the same pretrained weights
    assert torch.equal(trainer.teacher.vit.blocks[0].attn.qkv.weight,
                       trainer.student.vit.blocks[0].attn.qkv.weight)
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    vit_ids = {id(p) for p in trainer.student.vit.parameters()}
    assert all(id(p) in vit_ids for p in groups[1]["params"])
    assert not any(id(p) in vit_ids for p in groups[0]["params"])
    base = float(cfg.train.lr)
    trainer._apply_warmup(4)                                  # half-way through warmup
    assert groups[0]["lr"] == pytest.approx(base * 0.5)
    assert groups[1]["lr"] == pytest.approx(base * 0.5 * 0.1)
    trainer._apply_warmup(100)
    assert groups[1]["lr"] == pytest.approx(base * 0.1)


def test_no_pretrained_keeps_single_param_group(tmp_path):
    """Without a checkpoint the optimizer is unchanged, so older runs still resume."""
    trainer = SSLTrainer(small_config(), output_dir=tmp_path, device="cpu")
    assert len(trainer.optimizer.param_groups) == 1


# --------------------------------------------------------------------------- 5. harmonise
def _obs(baseline, day="2023-07-15"):
    return Observation(glims_id="G", observation_id=1, year=int(day[:4]), season="JJA",
                       acquisition_date=day, band_paths={}, scl_path=None, cloud_pct=0.0,
                       processing_baseline=baseline)


def test_baseline_offset_follows_recorded_baseline():
    pre = Preprocessor(small_config(data={"harmonize_baseline": True}))
    assert pre.baseline_offset(_obs("05.10")) == 1000.0
    assert pre.baseline_offset(_obs("04.00", "2021-12-27")) == 1000.0   # date rule would miss
    assert pre.baseline_offset(_obs("02.12", "2019-07-15")) == 0.0
    assert pre.baseline_offset(_obs("03.00", "2016-07-15")) == 0.0
    # unrecorded -> ESA switch-over date fallback
    assert pre.baseline_offset(_obs(None, "2022-03-01")) == 1000.0
    assert pre.baseline_offset(_obs(None, "2021-06-01")) == 0.0


def test_harmonisation_keeps_nodata_and_clips_dark_pixels():
    pre = Preprocessor(small_config(data={"harmonize_baseline": True}))
    raw = np.array([[[0.0, 1500.0, 900.0]]] * 6)        # (6,1,3): nodata, valid, dark valid
    valid = pre.validity_mask(raw, None)                 # nodata detected on RAW values
    out = pre.harmonize_dn(raw, _obs("05.10"))
    assert valid.tolist() == [[False, True, True]]
    assert out[0, 0].tolist() == [0.0, 500.0, 0.0]       # offset removed, clipped at 0
    assert np.array_equal(pre.harmonize_dn(raw, _obs("02.12")), raw)


def test_older_configs_never_harmonise():
    """A checkpoint trained before the fix must keep its exact preprocessing."""
    cfg = small_config()
    cfg._data["data"].pop("harmonize_baseline", None)
    pre = Preprocessor(cfg)
    assert pre.baseline_offset(_obs("05.10")) == 0.0


def test_manifests_without_the_field_still_load(tmp_path):
    import json
    from glacier_ssl.data.manifest import load_manifest
    o = {"glims_id": "G", "observation_id": 1, "year": 2020, "season": "JJA",
         "acquisition_date": "2020-07-15", "band_paths": {}, "scl_path": None,
         "cloud_pct": 0.0}
    payload = {"glaciers": {"G": {"glims_id": "G", "region": "r", "split": "train",
                                  "area_km2": 1.0, "centroid": [0, 0], "observations": [o]}}}
    (tmp_path / "m.json").write_text(json.dumps(payload))
    seqs = load_manifest(tmp_path / "m.json")
    assert seqs["G"].observations[0].processing_baseline is None


def test_pretrained_baseline_differs_from_random_only_in_the_vit(tmp_path):
    """The two eval baselines isolate one variable: the pretrained spatial encoder."""
    from glacier_ssl.eval.extract import build_backbones
    path = tmp_path / "pre.pt"
    torch.save(_prithvi_like_state(MultispectralViT(small_config())), path)
    cfg = small_config(vit={"pretrained_ckpt": str(path)})
    bbs = build_backbones(cfg, None, ["random", "pretrained"], torch.device("cpu"))
    rnd, pre = bbs["random"], bbs["pretrained"]
    state = torch.load(path)
    assert torch.equal(pre.vit.blocks[0].attn.qkv.weight,
                       state["encoder.blocks.0.attn.qkv.weight"])
    assert not torch.equal(rnd.vit.blocks[0].attn.qkv.weight,
                           pre.vit.blocks[0].attn.qkv.weight)
    for a, b in zip(rnd.tet.parameters(), pre.tet.parameters()):
        assert torch.equal(a, b)                 # identical untrained TET
    for a, b in zip(rnd.hpe.parameters(), pre.hpe.parameters()):
        assert torch.equal(a, b)                 # identical untrained HPE
