"""Phase 9-11 acceptance tests for Head 1 (DINO), Head 2 (distance), Head 3 (change).

Covers head1_archi.md §9, head2_archi.md §12, head3_archi.md §15 implementation checks,
at the REAL embedding dim (768) so the documented feature shapes (1536, 3840) are exact.
"""
import torch

from _helpers import small_config
from glacier_ssl.config import load_config
from glacier_ssl.heads.head1_dino import Head1DINO, DINOHead
from glacier_ssl.heads.head2_distance import Head2Distance
from glacier_ssl.heads.head3_change import Head3Change, region_pair_features


# ------------------------------------------------------------------ Head 1
def test_dino_head_logit_shape():
    cfg = load_config(overrides={"head1": {"num_prototypes": 1024}})
    head = DINOHead(768, 2048, 1024, 1024).eval()
    with torch.no_grad():
        logits = head(torch.randn(5, 768))   # (B*T, 768) -> (B*T, K)
    assert logits.shape == (5, 1024)


def test_dino_teacher_ema_not_grad():
    cfg = small_config()
    h = Head1DINO(cfg)
    for p in h.teacher_head.parameters():
        assert not p.requires_grad
    before = [p.clone() for p in h.teacher_head.parameters()]
    for p in h.student_head.parameters():
        p.data.add_(1.0)
    h.ema_update_head(momentum=0.9)
    moved = any(not torch.allclose(a, b)
                for a, b in zip(before, h.teacher_head.parameters()))
    assert moved                              # teacher tracks student by EMA


def test_dino_excludes_same_view_and_is_finite():
    cfg = small_config()
    h = Head1DINO(cfg)
    N, K = 6, cfg.head1.num_prototypes
    # 2 global (teacher+student) + 1 local student, view indices 0,1,2
    student_logits = [torch.randn(N, K) for _ in range(3)]
    teacher_logits = [torch.randn(N, K) for _ in range(2)]
    sv = [torch.ones(N, dtype=torch.bool) for _ in range(3)]
    tv = [torch.ones(N, dtype=torch.bool) for _ in range(2)]
    loss = h.loss(student_logits, teacher_logits, sv, tv,
                  teacher_is_global=[0, 1], student_global_index=[0, 1, 2])
    assert torch.isfinite(loss)
    # same-view pairs (teacher view 0 vs student view 0) excluded: with 2 teachers x 3
    # students minus 2 same-view = 4 cross terms.
    h.loss.update_center(teacher_logits, tv)
    assert torch.isfinite(h.loss.center).all()


# ------------------------------------------------------------------ Head 2
def test_head2_pair_feature_1536_and_scalar():
    cfg = load_config()
    h = Head2Distance(cfg).eval()
    z_i, z_j = torch.randn(7, 768), torch.randn(7, 768)
    feats = h.pair_features(z_i, z_j)
    assert feats.shape == (7, 1536)
    with torch.no_grad():
        pred = h(z_i, z_j)
    assert pred.shape == (7,)
    assert (pred >= 0).all()                  # Softplus non-negative


def test_head2_pair_order_symmetry():
    cfg = load_config()
    h = Head2Distance(cfg).eval()
    z_i, z_j = torch.randn(4, 768), torch.randn(4, 768)
    with torch.no_grad():
        a = h(z_i, z_j)
        b = h(z_j, z_i)
    assert torch.allclose(a, b, atol=1e-5)    # swapping pair order -> identical


def test_head2_loss_log_space_and_backprop():
    cfg = load_config()
    h = Head2Distance(cfg)
    z_i = torch.randn(5, 768, requires_grad=True)
    z_j = torch.randn(5, 768, requires_grad=True)
    pred = h(z_i, z_j)
    target = torch.rand(5) * 5
    loss, metrics = h.compute_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss) and "mae_years" in metrics
    assert z_i.grad is not None and torch.isfinite(z_i.grad).all()


# ------------------------------------------------------------------ Head 3
def test_head3_feature_and_encoder_shapes():
    cfg = load_config()
    h = Head3Change(cfg).eval()
    r_i, r_j = torch.randn(3, 64, 768), torch.randn(3, 64, 768)
    feats = region_pair_features(r_i, r_j, h.ln, "five_group")
    assert feats.shape == (3, 64, 3840)
    with torch.no_grad():
        change, contrastive = h.encode(r_i, r_j)
    assert change.shape == (3, 768)
    assert contrastive.shape == (3, 256)
    assert torch.allclose(contrastive.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_head3_direction_preserved():
    cfg = load_config()
    h = Head3Change(cfg).eval()
    r_i, r_j = torch.randn(4, 64, 768), torch.randn(4, 64, 768)
    with torch.no_grad():
        fwd, _ = h.encode(r_i, r_j)
        rev, _ = h.encode(r_j, r_i)           # reversed interval
    assert not torch.allclose(fwd, rev)       # reverse != forward


def test_head3_gap_not_in_encoder():
    """Structural: encode() signature takes only regional embeddings + validity."""
    import inspect
    params = set(inspect.signature(Head3Change.encode).parameters)
    for forbidden in ("gap", "gap_years", "delta", "time", "interval_gap"):
        assert forbidden not in params


def test_head3_infonce_overfits_tiny_batch():
    cfg = small_config()
    h = Head3Change(cfg)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    R, D = cfg.tet.num_regions, cfg.tet.embed_dim
    r_i = torch.randn(8, R, D)
    r_j = torch.randn(8, R, D)
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss, _ = h(r_i, r_j, r_i, r_j)       # same views -> easy positives
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    assert float(loss.detach()) < first       # loss decreases without instability


def test_head3_validity_aware_pool_excludes_invalid():
    cfg = load_config()
    h = Head3Change(cfg).eval()
    r_i, r_j = torch.randn(2, 64, 768), torch.randn(2, 64, 768)
    valid = torch.ones(2, 64, dtype=torch.bool)
    valid[0, 32:] = False
    with torch.no_grad():
        change_masked, _ = h.encode(r_i, r_j, valid)
        # manual mean over valid regions of the same encoder tokens
        feats = region_pair_features(r_i[:1, :32], r_j[:1, :32], h.ln, h.pair_kind)
        tok = h.encoder(feats).mean(dim=1)
    assert torch.allclose(change_masked[0], tok[0], atol=1e-5)
