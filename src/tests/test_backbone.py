"""Phase 5-8 acceptance tests: ViT, HPE, TET, and the integrated backbone.

Covers the implementation-check lists in VIT_aarchi.md §10, hpe.md §11, Temptrans_aarchi.md
minimum tests, plus variable-length and mask-inertness (build plan Phase 8).
"""
import torch

from _helpers import small_config
from glacier_ssl.backbone.vit import MultispectralViT
from glacier_ssl.backbone.hpe import HierarchicalPatchEmbedding
from glacier_ssl.backbone.tet import TemporalEvolutionTransformer, seasons_to_ids
from glacier_ssl.backbone.backbone import GlacierBackbone
from glacier_ssl.config import load_config


# ---------------------------------------------------------------- ViT (Phase 5)
def test_vit_token_count_and_dim():
    cfg = small_config()
    vit = MultispectralViT(cfg).eval()
    x = torch.randn(3, 6, 128, 128)
    with torch.no_grad():
        out = vit(x)
    grid = 128 // 16
    assert out.shape == (3, grid * grid, 64)   # (B, tokens, dim), no CLS/pool


def test_vit_real_contract_1024_tokens():
    """Real dims: 512x512, patch 16 -> exactly 1024 tokens of dim 768 (depth=1 for speed)."""
    cfg = load_config(overrides={"vit": {"depth": 1, "drop_path": 0.0}})
    vit = MultispectralViT(cfg).eval()
    x = torch.randn(1, 6, 512, 512)
    with torch.no_grad():
        out = vit(x)
    assert out.shape == (1, 1024, 768)


def test_vit_shared_weights_same_every_timestamp():
    cfg = small_config()
    vit = MultispectralViT(cfg).eval()
    x = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        a, b = vit(x), vit(x)
    assert torch.allclose(a, b)                # deterministic, shared weights


# ---------------------------------------------------------------- HPE (Phase 6)
def test_hpe_two_stage_reduction():
    cfg = load_config(overrides={"hpe": {"embed_dim": 768, "input_grid": 32}})
    hpe = HierarchicalPatchEmbedding(cfg).eval()
    x = torch.randn(5, 1024, 768)              # T folded into batch
    with torch.no_grad():
        out = hpe(x)
    assert out.shape == (5, 64, 768)           # 8x8 regional grid
    assert hpe.num_regions == 64


def test_hpe_variable_T_including_single():
    cfg = load_config(overrides={"hpe": {"embed_dim": 768, "input_grid": 32}})
    hpe = HierarchicalPatchEmbedding(cfg).eval()
    for T in (1, 2, 7):
        with torch.no_grad():
            out = hpe(torch.randn(T, 1024, 768))
        assert out.shape == (T, 64, 768)


def test_hpe_no_cross_timestamp_mixing():
    """Changing timestamp t must not change any other timestamp's output."""
    cfg = load_config(overrides={"hpe": {"embed_dim": 768, "input_grid": 32}})
    hpe = HierarchicalPatchEmbedding(cfg).eval()
    x = torch.randn(4, 1024, 768)
    with torch.no_grad():
        out1 = hpe(x)
        x2 = x.clone()
        x2[2] = torch.randn(1024, 768)         # perturb only timestamp 2
        out2 = hpe(x2)
    assert torch.allclose(out1[[0, 1, 3]], out2[[0, 1, 3]])
    assert not torch.allclose(out1[2], out2[2])


# ---------------------------------------------------------------- TET (Phase 7)
def _tet_inputs(cfg, B=2, T=5, R=4):
    regional = torch.randn(B, T, R, cfg.tet.embed_dim)
    season_ids = torch.randint(0, 4, (B, T))
    elapsed = torch.rand(B, T) * 5
    span = torch.rand(B)
    tmask = torch.ones(B, T, dtype=torch.bool)
    return regional, season_ids, elapsed, span, tmask


def test_tet_output_shapes_broadcast_mean():
    cfg = small_config()
    tet = TemporalEvolutionTransformer(cfg).eval()
    regional, s, e, sp, m = _tet_inputs(cfg)
    with torch.no_grad():
        out = tet(regional, s, e, sp, m)
    assert out["regional_out"].shape == (2, 5, 4, 64)
    assert out["timestamp_out"].shape == (2, 5, 64)
    assert out["sequence_out"].shape == (2, 64)


def test_tet_cross_attention_mode_shapes():
    cfg = small_config(tet={"context_mode": "cross_attention"})
    tet = TemporalEvolutionTransformer(cfg).eval()
    regional, s, e, sp, m = _tet_inputs(cfg)
    with torch.no_grad():
        out = tet(regional, s, e, sp, m)
    assert out["sequence_out"].shape == (2, 64)
    assert out["timestamp_out"].shape == (2, 5, 64)


def test_tet_context_token_actually_updates():
    """sequence_out must differ from the raw init token -> context communicates."""
    cfg = small_config()
    tet = TemporalEvolutionTransformer(cfg).eval()
    regional, s, e, sp, m = _tet_inputs(cfg)
    with torch.no_grad():
        out = tet(regional, s, e, sp, m)
        raw = tet.sequence_token.expand(2, 64) + tet.span_encoding(sp)
    assert not torch.allclose(out["sequence_out"], tet.context_norm(raw))


def test_tet_padding_is_inert():
    """Appending padded timesteps must not change valid-timestep outputs."""
    cfg = small_config()
    tet = TemporalEvolutionTransformer(cfg).eval()
    B, T, R, D = 2, 4, 4, cfg.tet.embed_dim
    regional = torch.randn(B, T, R, D)
    s = torch.randint(0, 4, (B, T))
    e = torch.rand(B, T) * 3
    sp = torch.rand(B)
    m = torch.ones(B, T, dtype=torch.bool)
    with torch.no_grad():
        base = tet(regional, s, e, sp, m)
    # pad with 3 junk timesteps marked invalid
    pad = 3
    regional_p = torch.cat([regional, torch.randn(B, pad, R, D)], dim=1)
    s_p = torch.cat([s, torch.randint(0, 4, (B, pad))], dim=1)
    e_p = torch.cat([e, torch.rand(B, pad) * 99], dim=1)
    m_p = torch.cat([m, torch.zeros(B, pad, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        padded = tet(regional_p, s_p, e_p, sp, m_p)
    assert torch.allclose(base["timestamp_out"], padded["timestamp_out"][:, :T], atol=1e-5)
    assert torch.allclose(base["sequence_out"], padded["sequence_out"], atol=1e-5)
    assert torch.allclose(base["regional_out"], padded["regional_out"][:, :T], atol=1e-5)


def test_tet_timestamp_pool_is_masked_mean():
    cfg = small_config()
    tet = TemporalEvolutionTransformer(cfg).eval()
    regional, s, e, sp, m = _tet_inputs(cfg, R=4)
    rm = torch.ones(2, 5, 4, dtype=torch.bool)
    rm[0, 0, 2:] = False                        # mark 2 regions invalid at one step
    with torch.no_grad():
        out = tet(regional, s, e, sp, m, region_mask=rm)
        manual = out["regional_out"][0, 0, :2].mean(dim=0)
    assert torch.allclose(out["timestamp_out"][0, 0], manual, atol=1e-5)


# ---------------------------------------------------- integrated backbone (Phase 8)
def test_backbone_end_to_end_shapes():
    cfg = small_config()
    bb = GlacierBackbone(cfg).eval()
    B, T = 2, 3
    imgs = torch.randn(B, T, 6, 128, 128)
    m = torch.ones(B, T, dtype=torch.bool)
    e = torch.rand(B, T) * 4
    sp = torch.rand(B)
    seasons = [["DJF", "MAM", "JJA"], ["SON", "DJF", "MAM"]]
    with torch.no_grad():
        out = bb(imgs, m, e, sp, seasons)
    assert out["regional_out"].shape == (B, T, 4, 64)
    assert out["timestamp_out"].shape == (B, T, 64)
    assert out["sequence_out"].shape == (B, 64)


def test_backbone_mixed_lengths_padding_inert():
    """A batch mixing short/long sequences: padded tail must not affect the short seq."""
    cfg = small_config()
    bb = GlacierBackbone(cfg).eval()
    imgs = torch.randn(2, 5, 6, 128, 128)
    m = torch.ones(2, 5, dtype=torch.bool)
    m[0, 3:] = False                            # sample 0 has length 3
    imgs[0, 3:] = 0
    e = torch.rand(2, 5) * 4
    sp = torch.rand(2)
    seasons = [["DJF", "MAM", "JJA", "SON", "DJF"], ["SON", "DJF", "MAM", "JJA", "SON"]]
    with torch.no_grad():
        full = bb(imgs, m, e, sp, seasons)
        # re-run sample 0 alone at its true length 3
        out0 = bb(imgs[:1, :3], m[:1, :3], e[:1, :3], sp[:1], [seasons[0][:3]])
    assert torch.allclose(full["timestamp_out"][0, :3], out0["timestamp_out"][0],
                          atol=1e-5)
    assert torch.allclose(full["sequence_out"][0], out0["sequence_out"][0], atol=1e-5)


def test_seasons_to_ids():
    ids = seasons_to_ids([["DJF", "MAM"], ["JJA", "SON"]], 2, torch.device("cpu"))
    assert ids.tolist() == [[0, 1], [2, 3]]
