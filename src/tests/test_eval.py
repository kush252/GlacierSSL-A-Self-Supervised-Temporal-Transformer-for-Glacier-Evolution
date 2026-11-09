"""Tests for the post-training evaluation suite (glacier_ssl/eval + scripts/evaluate.py).

A miniature backbone (tests/_helpers.small_config) and synthetic dated sequences exercise the
real extraction + test code on CPU. The two properties the robustness claims rest on are
checked explicitly: cached features equal a direct backbone forward, and dropping timestamps
then re-running only the TET equals running the backbone on the shortened sequence.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from _helpers import small_config
from glacier_ssl.backbone.backbone import GlacierBackbone
from glacier_ssl.backbone.tet import seasons_to_ids
from glacier_ssl.data.manifest import GlacierSequence, Observation
from glacier_ssl.eval import (context, embedding_eval, extract, head2_eval, retrieval_eval,
                              robustness_eval, stats, trajectory_eval)
from glacier_ssl.eval.store import ResultsStore
from glacier_ssl.heads.head2_distance import Head2Distance

DEV = torch.device("cpu")
SEASONS = ["DJF", "MAM", "JJA", "SON"]
MONTHS = [1, 4, 7, 10]


def _sequence(gid: str, region: str, T: int = 8) -> GlacierSequence:
    obs = []
    for t in range(T):
        year = 2016 + t // 4
        obs.append(Observation(
            glims_id=gid, observation_id=1000 + t, year=year, season=SEASONS[t % 4],
            acquisition_date=f"{year}-{MONTHS[t % 4]:02d}-15", band_paths={}, scl_path=None,
            cloud_pct=float(5 * t)))
    return GlacierSequence(glims_id=gid, region=region, split="test", area_km2=1.0 + len(gid),
                           centroid=(7.0, 46.0 + len(gid) * 0.1), observations=obs)


def _items(cfg, n_glaciers: int = 10, T: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    S = int(cfg.data.image_size)
    items = []
    for k in range(n_glaciers):
        seq = _sequence(f"G{k:03d}E{k}N", "A" if k % 2 else "B", T)
        img = torch.randn(T, 6, S, S, generator=g)
        valid = torch.ones(T, S, S, dtype=torch.bool)
        items.append(extract.make_item(seq, seq.observations, img, valid))
    return items


@pytest.fixture(scope="module")
def world():
    """Backbones, extracted features for student/random, and a bundle with a Head 2."""
    cfg = small_config()
    torch.manual_seed(0)
    student = GlacierBackbone(cfg).eval()
    head2 = Head2Distance(cfg).eval()
    bundle = {"cfg": cfg, "step": 7, "epoch": 0, "student": student.state_dict(),
              "teacher": student.state_dict(), "head2": head2.state_dict()}
    bbs = extract.build_backbones(cfg, bundle, ["student", "random"], DEV)
    items = _items(cfg)
    meta, feats = extract.extract_all(bbs, items, DEV, cfg, keep_tokens={"student"})
    return {"cfg": cfg, "bundle": bundle, "bbs": bbs, "items": items,
            "fs": extract.FeatureSet("test", meta, feats)}


def make_ctx(world, tmp_path, **kw):
    store = ResultsStore(tmp_path, "run", manifest={"checkpoint": {"path": "x", "step": 7}})
    return context.EvalContext(
        cfg=world["cfg"], store=store, device=DEV, project_root=tmp_path, primary="student",
        test=world["fs"], train=None, bundle=world["bundle"], seed=0, baseline="random",
        extra={"occlusion_glaciers": 0}, **kw)


# --------------------------------------------------------------------------- store/stats
def test_store_writes_tables_index_and_report(tmp_path):
    store = ResultsStore(tmp_path, "r1", manifest={"checkpoint": {"path": "c.pt", "step": 3},
                                                   "args": {"backbone": "student"}})
    md = store.write_table("t/a", [{"x": 1, "y": float("nan")}, {"x": 2, "y": 0.5}])
    assert (store.dir / "t/a.csv").exists() and (store.dir / "t/a.md").exists()
    assert "| x | y |" in md
    store.record("head2", {"mae": float("nan"), "n": np.int64(4)})
    store.add_section("S", "body")
    store.finalize()
    rep = (store.dir / "REPORT.md").read_text(encoding="utf-8")
    assert "## S" in rep and "body" in rep
    summary = json.loads((store.dir / "summary.json").read_text())
    assert summary["tests"]["head2"]["headline"]["mae"] is None       # NaN -> null
    lines = (tmp_path / "results_index.jsonl").read_text().strip().splitlines()
    assert json.loads(lines[0])["test"] == "head2"


def test_stats_helpers():
    a = np.array([1, 1, 2, 2, 3, 3.0])
    assert stats.spearman(a, a) == pytest.approx(1.0)
    assert np.isnan(stats.spearman(np.ones(6), a))
    rng = np.random.default_rng(0)
    collapsed = np.tile(rng.normal(size=(1, 16)), (50, 1)) + 1e-9 * rng.normal(size=(50, 16))
    spread = rng.normal(size=(200, 16))
    assert stats.effective_rank(collapsed) < 2.0
    assert stats.effective_rank(spread) > 12.0
    p, lo, hi = stats.cluster_bootstrap_ratio(np.arange(1, 21.0), np.ones(20), 200, 0)
    assert lo <= p <= hi
    # partial correlation removes a shared confounder
    c = rng.normal(size=300)
    x, y = c + 0.1 * rng.normal(size=300), c + 0.1 * rng.normal(size=300)
    assert stats.spearman(x, y) > 0.8
    assert abs(stats.partial_spearman(x, y, c)) < 0.3


def test_occlusion_zeroes_exact_fraction():
    img = torch.ones(3, 6, 64, 64)
    valid = torch.ones(3, 64, 64, dtype=torch.bool)
    out, v = extract.occlude(img, valid, 0.25, seed=1)
    frac = (~v).float().mean(dim=(1, 2))
    assert torch.allclose(frac, torch.full((3,), 0.25))
    assert float(out[~v.unsqueeze(1).expand_as(out)].abs().sum()) == 0.0


# --------------------------------------------------------------------------- extraction
def test_cached_features_equal_direct_forward(world):
    item, bb = world["items"][0], world["bbs"]["student"]
    m = item["meta"]
    T = len(m["dates"])
    imgs = item["imgs"].unsqueeze(0)
    el = torch.as_tensor(m["elapsed"])[None]
    out = bb(imgs, torch.ones(1, T, dtype=torch.bool), el, el[:, -1] - el[:, 0],
             seasons_to_ids([m["seasons"]], T, DEV))
    cached = world["fs"].feats["student"][m["glims_id"]]
    assert np.allclose(cached["timestamp_out"], out["timestamp_out"][0].detach().numpy(),
                       atol=1e-5)
    assert np.allclose(cached["sequence_out"], out["sequence_out"][0].detach().numpy(),
                       atol=1e-5)
    assert cached["visual_pool"].shape == (T, bb.embed_dim)
    assert world["fs"].feats["random"][m["glims_id"]]["tokens"] is None


def test_chunked_encoding_is_exact(world):
    bb, imgs = world["bbs"]["student"], world["items"][1]["imgs"].unsqueeze(0)
    amp = (False, torch.float16)
    a = extract.encode_tokens(bb, imgs, DEV, amp, chunk=64)
    b = extract.encode_tokens(bb, imgs, DEV, amp, chunk=3)
    assert torch.allclose(a, b, atol=1e-6)


def test_dropping_timestamps_matches_shorter_input(world):
    """The foundation of the robustness test: re-running only the TET on a subset of cached
    tokens is what the backbone would produce for the shortened sequence."""
    gid = world["items"][2]["meta"]["glims_id"]
    m = world["fs"].meta[gid]
    bb = world["bbs"]["student"]
    tokens = world["fs"].feats["student"][gid]["tokens"]
    sel = np.array([0, 2, 3, 6])
    z_fast, s_fast = robustness_eval._tet(bb, tokens, m, sel, DEV, (False, torch.float16))

    imgs = world["items"][2]["imgs"][sel].unsqueeze(0)
    el = torch.as_tensor(np.asarray(m["elapsed"])[sel] - m["elapsed"][sel[0]])[None].float()
    out = bb(imgs, torch.ones(1, len(sel), dtype=torch.bool), el, el[:, -1] - el[:, 0],
             seasons_to_ids([[m["seasons"][k] for k in sel]], len(sel), DEV))
    z_ref = out["timestamp_out"][0].detach().numpy()
    assert stats.cosine_dist_rows(z_fast, z_ref).max() < 1e-4
    assert stats.cosine_dist_rows(s_fast[None], out["sequence_out"].detach().numpy())[0] < 1e-4


# --------------------------------------------------------------------------- pairs
def test_pair_builder_uses_dates_and_skips_unusable():
    rec = {"gid": "g", "elapsed": np.array([0.0, 0.25, 0.5, 0.5, 1.0]),
           "seasons": ["DJF", "MAM", "JJA", "JJA", "SON"],
           "usable": np.array([True, True, True, True, False])}
    P = head2_eval.build_pairs([rec], 100, np.random.default_rng(0))
    assert (P["i"] < P["j"]).all()
    assert 4 not in P["i"] and 4 not in P["j"]                     # unusable excluded
    assert (P["gap"] > 0).all()                                    # duplicate-date pair dropped
    pairs = {(int(i), int(j)): g for i, j, g in zip(P["i"], P["j"], P["gap"])}
    assert pairs[(0, 2)] == pytest.approx(0.5)
    assert (2, 3) not in pairs
    capped = head2_eval.build_pairs([rec], 2, np.random.default_rng(0))
    assert len(capped["gap"]) == 2


# --------------------------------------------------------------------------- tests e2e
def test_head2_eval_end_to_end(world, tmp_path, monkeypatch):
    monkeypatch.setattr(head2_eval, "metadata_baselines", lambda *a, **k: {})
    ctx = make_ctx(world, tmp_path)
    out = head2_eval.run(ctx)
    assert out["n_pairs"] > 50 and np.isfinite(out["mae_years"])
    assert out["pair_order_max_abs_diff"] < 1e-5                    # Experiment G
    d = ctx.store.dir / "head2"
    assert (d / "metrics.json").exists() and (d / "mae_by_gap_range.csv").exists()
    assert (d / "fig_calibration_error_head2.png").exists()


def test_label_free_tests_run_and_write(world, tmp_path):
    ctx = make_ctx(world, tmp_path)
    emb = embedding_eval.run(ctx)
    assert "student" in emb and "random" in emb
    assert emb["student"]["effective_rank"] > 1.0
    traj = trajectory_eval.run(ctx)
    assert traj["student"]["n_glaciers"] == len(world["fs"].meta)
    ret = retrieval_eval.run(ctx)
    ident = ret["student"]["identity"]
    assert 0.0 <= ident["P@1"] <= 1.0 and ident["chance_P@1"] == pytest.approx(7 / 79)
    for sub in ("embeddings", "trajectories", "retrieval"):
        assert (ctx.store.dir / sub / "metrics.json").exists()
    assert (ctx.store.dir / "embeddings" / "fig_tsne.png").exists()


def test_retrieval_perfect_embedding_scores_one():
    g = np.repeat(np.arange(5), 4)
    X = np.eye(5)[g] + 0.01 * np.random.default_rng(0).normal(size=(20, 5))
    out = retrieval_eval.identity_task(retrieval_eval.sim_matrix(X), g)
    assert out["P@1"] == 1.0 and out["mAP"] > 0.99


def test_robustness_run(world, tmp_path):
    ctx = make_ctx(world, tmp_path)
    out = robustness_eval.run(ctx)
    drops = out["missing_timestamps"]
    assert drops["dropped 0% (full)"]["drift"] == 0.0
    assert drops["dropped 50%"]["drift"] > 0.0           # context removal moves embeddings
    assert "L=8" in out["sequence_length"] and "L=2" in out["sequence_length"]
    assert set(out["reidentification"]) <= {"2", "3", "4"}   # needs n >= 2L with T=8
    assert (ctx.store.dir / "robustness" / "fig_missing_and_length.png").exists()


def test_run_report_and_index_roundtrip(world, tmp_path):
    ctx = make_ctx(world, tmp_path)
    retrieval_eval.run(ctx)
    ctx.store.finalize()
    rep = (ctx.store.dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Test 4 - Retrieval" in rep and "![](retrieval/fig_neighbours.png)" in rep
