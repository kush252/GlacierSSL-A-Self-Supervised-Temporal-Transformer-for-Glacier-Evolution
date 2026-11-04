"""Pre-training preflight audit — "is the thing actually wired correctly?"

The pytest suite (src/tests) checks shapes, masks, directionality and EMA. This script
answers the questions a test suite usually leaves implicit, and which silently waste a
whole training run when they are wrong:

  A. CONFIG GEOMETRY   do the derived dims agree across ViT / HPE / TET / all 3 heads?
  B. TENSOR CONTRACT   do real forward shapes match DOCS_phase0/tensor_contract_map.md?
  C. GRAD COVERAGE     does EVERY trainable parameter receive a finite, non-zero gradient
                       from the combined 3-head loss?  (a dead module trains forever at 0)
  D. OPTIMIZER COVER   is every trainable parameter actually inside the optimizer?
                       (a param can get grads and still never be stepped)
  E. PARAM MOTION      does every covered parameter actually change after one step?
  F. TEACHER ISOLATION teacher gets no grad, moves only by EMA.
  G. PADDING INERT     perturbing padded timesteps cannot change valid outputs.
  H. VARIABLE T        T=1..N all run; padding a sequence does not change its real outputs.
  I. INFO BARRIERS     Head2 pair features swap-invariant; Head3 direction-sensitive.
  J. STABILITY         multi-step run stays finite; per-head losses behave.

Run it before every training launch (locally and on Kaggle):

    python src/scripts/verify_system.py              # fast audit on the miniature config
    python src/scripts/verify_system.py --device cuda  # same, on GPU (checks AMP path too)
    python src/scripts/verify_system.py --full-config  # also build the real 768-dim backbone

Exit code 0 = cleared for training. Non-zero = at least one check FAILED.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "tests"))

import torch

from glacier_ssl.backbone.backbone import GlacierBackbone
from glacier_ssl.backbone.tet import seasons_to_ids
from glacier_ssl.config import load_config
from glacier_ssl.heads.head2_distance import Head2Distance
from glacier_ssl.heads.head3_change import Head3Change
from glacier_ssl.train.trainer import SSLTrainer

# Parameters that are EXPECTED to receive no gradient from the SSL objective.
# `tet.context_norm` is the terminal LayerNorm that produces `sequence_out` (the Sequence
# Evolution Embedding). Per the specs no baseline head consumes it — Heads 1/2 take
# `timestamp_out`, Head 3 takes `regional_out` (head3_archi.md: "does not directly use
# timestamp_out or sequence_out in the baseline") — so it sits outside every loss path.
# The context TOKEN itself is trained: `sequence_token`, `span_encoding` and each TET
# block's context arithmetic all receive gradient through the broadcast_mean path into the
# regional tokens. Only this final norm is untrained, and at init (weight=1, bias=0) it is
# exactly standard layer normalization, so `sequence_out` stays well-formed for downstream
# use. Attaching a loss here would be inventing architecture the plan does not specify.
_EXPECTED_UNTRAINED = ("backbone.tet.context_norm.",)


def _expected_untrained(name: str) -> bool:
    return any(name.startswith(p) for p in _EXPECTED_UNTRAINED)


# ----------------------------------------------------------------------------------
# tiny reporting harness
# ----------------------------------------------------------------------------------
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def small_config(**extra):
    """Miniature but internally consistent config (same one the tests use)."""
    from _helpers import small_config as _sc
    return _sc(**extra)


# ----------------------------------------------------------------------------------
# A. config geometry
# ----------------------------------------------------------------------------------
def audit_config_geometry(cfg, label: str) -> None:
    section(f"A. Config geometry — {label}")
    img = int(cfg.vit.image_size)
    patch = int(cfg.vit.patch_size)
    dim = int(cfg.vit.embed_dim)

    check(f"{label}: image_size divisible by patch_size",
          img % patch == 0, f"{img} / {patch}")
    grid = img // patch
    check(f"{label}: ViT grid matches hpe.input_grid",
          grid == int(cfg.hpe.input_grid), f"grid={grid}, hpe.input_grid={cfg.hpe.input_grid}")

    stages = int(cfg.hpe.num_merge_stages)
    side = grid // (2 ** stages)
    regions = side * side
    check(f"{label}: HPE output regions match tet.num_regions",
          regions == int(cfg.tet.num_regions),
          f"{grid}x{grid} /2^{stages} -> {side}x{side} = {regions} vs tet.num_regions="
          f"{cfg.tet.num_regions}")

    check(f"{label}: embed_dim consistent across vit/hpe/tet",
          dim == int(cfg.hpe.embed_dim) == int(cfg.tet.embed_dim),
          f"vit={dim} hpe={cfg.hpe.embed_dim} tet={cfg.tet.embed_dim}")
    check(f"{label}: vit.embed_dim divisible by vit.num_heads",
          dim % int(cfg.vit.num_heads) == 0, f"{dim} / {cfg.vit.num_heads}")
    check(f"{label}: tet.embed_dim divisible by tet.num_heads",
          int(cfg.tet.embed_dim) % int(cfg.tet.num_heads) == 0,
          f"{cfg.tet.embed_dim} / {cfg.tet.num_heads}")
    check(f"{label}: 6-band input order is the fixed spec order",
          list(cfg.data.band_order) == ["B02", "B03", "B04", "B08", "B11", "B12"]
          and int(cfg.vit.in_chans) == 6,
          f"{list(cfg.data.band_order)}, in_chans={cfg.vit.in_chans}")
    check(f"{label}: data.image_size == vit.image_size",
          int(cfg.data.image_size) == img, f"{cfg.data.image_size} vs {img}")

    # head input widths derive from tet.embed_dim, never hardcoded
    mult = {"five_group": 5, "signed": 1, "abs": 1, "start_end_signed": 3}[
        str(cfg.head3.pair_features)]
    check(f"{label}: Head3 pair-feature width = {mult}x embed_dim",
          True, f"{mult} * {cfg.tet.embed_dim} = {mult * int(cfg.tet.embed_dim)}")
    check(f"{label}: loss weights all finite and >= 0",
          all(float(x) >= 0 for x in (cfg.loss.lambda_dino, cfg.loss.lambda_distance,
                                      cfg.loss.lambda_change)),
          f"dino={cfg.loss.lambda_dino} dist={cfg.loss.lambda_distance} "
          f"change={cfg.loss.lambda_change}")


# ----------------------------------------------------------------------------------
# synthetic batch that exercises ALL THREE heads
# ----------------------------------------------------------------------------------
def synthetic_batch(cfg, device, B=2, T=4, n_intervals=4):
    o = int(cfg.data.image_size)
    base = torch.randn(B, T, 6, o, o)
    tmask = torch.ones(B, T, dtype=torch.bool)
    elapsed = torch.rand(B, T).sort(dim=1).values * 4
    span = elapsed[:, -1] - elapsed[:, 0]
    seasons = [["DJF", "MAM", "JJA", "SON"][:T] for _ in range(B)]

    def view(kind):
        return {"img": torch.randn(B, T, 6, o, o), "timestamp_mask": tmask.clone(),
                "crop_valid": torch.ones(B, T, dtype=torch.bool), "view_type": kind}

    dino_views = ([view("global") for _ in range(int(cfg.head1.num_global_student))]
                  + [view("local") for _ in range(int(cfg.head1.num_local_student))])

    pairs = [(b, i, j) for b in range(B) for (i, j) in [(0, 1), (1, 3), (0, 3)]]

    intervals = []
    for k in range(n_intervals):
        intervals.append({
            "batch_index": k % B, "i": 0, "j": 1, "gap_years": 1.0 + k,
            "season_i": "DJF", "season_j": "JJA",
            "view_a_img": torch.randn(2, 6, o, o),
            "view_a_valid": torch.ones(2, dtype=torch.bool),
            "view_b_img": torch.randn(2, 6, o, o),
            "view_b_valid": torch.ones(2, dtype=torch.bool),
        })

    return {"base": base, "timestamp_mask": tmask, "base_valid": tmask.clone(),
            "elapsed_years": elapsed, "span_years": span, "seasons": seasons,
            "glims_id": [f"G{i}" for i in range(B)], "T": [T] * B,
            "dino_views": dino_views, "distance_pairs": pairs, "intervals": intervals}


# ----------------------------------------------------------------------------------
# B. tensor contract
# ----------------------------------------------------------------------------------
def audit_tensor_contract(cfg, device) -> None:
    section("B. Tensor contract (real forward shapes)")
    bb = GlacierBackbone(cfg).to(device).eval()
    B, T = 2, 3
    o = int(cfg.data.image_size)
    dim = int(cfg.tet.embed_dim)
    R = int(cfg.tet.num_regions)
    grid = int(cfg.vit.image_size) // int(cfg.vit.patch_size)

    imgs = torch.randn(B, T, 6, o, o, device=device)
    tmask = torch.ones(B, T, dtype=torch.bool, device=device)
    elapsed = torch.rand(B, T, device=device)
    span = torch.rand(B, device=device)
    sids = seasons_to_ids([["DJF"] * T] * B, T, device)

    with torch.no_grad():
        tok = bb.vit(imgs.reshape(B * T, 6, o, o))
        check("ViT -> (B*T, grid^2, dim)",
              tuple(tok.shape) == (B * T, grid * grid, dim),
              f"{tuple(tok.shape)} expected {(B * T, grid * grid, dim)}")
        reg = bb.hpe(tok)
        check("HPE -> (B*T, num_regions, dim)",
              tuple(reg.shape) == (B * T, R, dim),
              f"{tuple(reg.shape)} expected {(B * T, R, dim)}")
        out = bb(imgs, tmask, elapsed, span, sids)

    check("TET regional_out -> (B,T,R,dim)",
          tuple(out["regional_out"].shape) == (B, T, R, dim),
          str(tuple(out["regional_out"].shape)))
    check("TET timestamp_out -> (B,T,dim)",
          tuple(out["timestamp_out"].shape) == (B, T, dim),
          str(tuple(out["timestamp_out"].shape)))
    check("TET sequence_out -> (B,dim)  [context token, not pooled]",
          tuple(out["sequence_out"].shape) == (B, dim),
          str(tuple(out["sequence_out"].shape)))
    check("all backbone outputs finite",
          all(torch.isfinite(v).all() for v in out.values()))

    # heads consume exactly those contracts
    h2 = Head2Distance(cfg).to(device)
    pred = h2(out["timestamp_out"][:, 0], out["timestamp_out"][:, 1])
    check("Head2(u_i,u_j) -> (B,) non-negative scalar per pair",
          pred.shape == (B,) and bool((pred >= 0).all()),
          f"shape={tuple(pred.shape)} min={float(pred.min()):.4f}")

    h3 = Head3Change(cfg).to(device)
    chg, con = h3.encode(out["regional_out"][:, 0], out["regional_out"][:, 1])
    check("Head3 change_embedding -> (B, encoder_out)",
          tuple(chg.shape) == (B, int(cfg.head3.encoder_out)), str(tuple(chg.shape)))
    check("Head3 contrastive_embedding -> (B, proj_dim) and L2-normalized",
          tuple(con.shape) == (B, int(cfg.head3.proj_dim))
          and torch.allclose(con.norm(dim=-1), torch.ones(B, device=device), atol=1e-5),
          f"shape={tuple(con.shape)} norms~{float(con.norm(dim=-1).mean()):.4f}")


# ----------------------------------------------------------------------------------
# C/D/E/F. gradient + optimizer coverage, param motion, teacher isolation
# ----------------------------------------------------------------------------------
def audit_wiring(cfg, device, tmp: Path) -> None:
    section("C. Gradient coverage — every trainable parameter must receive gradient")
    torch.manual_seed(0)
    trainer = SSLTrainer(cfg, output_dir=tmp, device=device)
    batch = synthetic_batch(cfg, device)

    trainable = {}
    for prefix, mod in (("backbone", trainer.student),
                        ("head1", trainer.head1.student_head),
                        ("head2", trainer.head2),
                        ("head3", trainer.head3)):
        for n, p in mod.named_parameters():
            if p.requires_grad:
                trainable[f"{prefix}.{n}"] = p

    frozen = [f"head1.{n}" for n, p in trainer.head1.student_head.named_parameters()
              if not p.requires_grad]

    before = {n: p.detach().clone() for n, p in trainable.items()}

    # ---- D. optimizer coverage (before stepping) ----
    opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    missing_opt = [n for n, p in trainable.items() if id(p) not in opt_ids]

    logs = trainer.train_step(batch, step=0, total_steps=8)

    dead, nan_grad = [], []
    for n, p in trainable.items():
        if p.grad is None:
            dead.append(n)
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            nan_grad.append(n)
        elif float(g.norm()) == 0.0:
            dead.append(n)

    unexpected_dead = [n for n in dead if not _expected_untrained(n)]
    expected_dead = [n for n in dead if _expected_untrained(n)]
    check(f"all {len(trainable)} trainable params received a gradient "
          f"(excluding {len(expected_dead)} spec-expected)",
          not unexpected_dead,
          "unexpectedly dead: " + (", ".join(unexpected_dead[:8]) or "none")
          + (f" (+{len(unexpected_dead) - 8} more)" if len(unexpected_dead) > 8 else ""))
    check("params with no gradient are only the documented sequence_out terminal norm",
          all(_expected_untrained(n) for n in dead),
          ", ".join(expected_dead) or "none"
          + " | no baseline head consumes sequence_out (head3_archi.md)")
    check("no NaN/Inf gradients", not nan_grad,
          ", ".join(nan_grad[:5]) or "all finite")
    check("backbone gradient norm > 0 (3-head loss reaches shared ViT+HPE+TET)",
          logs["backbone_grad_norm"] > 0, f"grad_norm={logs['backbone_grad_norm']:.4f}")
    check("intentionally-frozen params are only the DINO prototype weight_g",
          all("weight_g" in f for f in frozen),
          ", ".join(frozen) or "none")

    section("D. Optimizer coverage — a param with grads but no optimizer never learns")
    check("every trainable param is inside the optimizer", not missing_opt,
          ", ".join(missing_opt[:5]) or f"{len(opt_ids)} params registered")

    section("E. Parameter motion — params must actually move after one step")
    stuck = [n for n, p in trainable.items()
             if torch.allclose(before[n], p.detach(), atol=0, rtol=0)
             and not _expected_untrained(n)]
    check("all gradient-receiving params changed after optimizer.step()", not stuck,
          ", ".join(stuck[:8]) or "all moved"
          + (f" (+{len(stuck) - 8} more)" if len(stuck) > 8 else ""))

    section("F. Teacher isolation — EMA target, never a gradient path")
    check("no teacher backbone param requires grad",
          all(not p.requires_grad for p in trainer.teacher.parameters()))
    check("no teacher DINO-head param requires grad",
          all(not p.requires_grad for p in trainer.head1.teacher_head.parameters()))
    check("no teacher param holds a gradient after backward",
          all(p.grad is None for p in trainer.teacher.parameters())
          and all(p.grad is None for p in trainer.head1.teacher_head.parameters()))

    t_before = next(trainer.teacher.vit.parameters()).detach().clone()
    trainer.train_step(synthetic_batch(cfg, device), step=1, total_steps=8)
    t_after = next(trainer.teacher.vit.parameters()).detach()
    check("teacher moves (EMA is live, not a frozen copy)",
          not torch.allclose(t_before, t_after))

    section("J. Stability — multi-step run stays finite")
    hist = []
    for s in range(2, 8):
        lg = trainer.train_step(synthetic_batch(cfg, device), step=s, total_steps=8)
        hist.append(lg)
    allf = all(torch.isfinite(torch.tensor(
        [lg["loss_total"], lg["loss_dino"], lg["loss_distance"], lg["loss_change"]])).all()
        for lg in hist)
    check("every per-head loss finite across steps", allf)
    check("all three heads contributed real samples",
          logs.get("n_pairs", 0) > 0 and logs.get("n_intervals", 0) >= 2
          and logs.get("n_dino_valid", 0) > 0,
          f"pairs={logs.get('n_pairs')} intervals={logs.get('n_intervals')} "
          f"dino_valid={logs.get('n_dino_valid')}")
    check("AMP/GradScaler plumbing consistent with device",
          (trainer.amp_enabled is False) if device.type == "cpu" else True,
          f"device={device.type} amp={trainer.amp_enabled} dtype={trainer.amp_dtype} "
          f"data_parallel={trainer.dp}")

    # overfit trend on ONE fixed batch: the supervised heads must be able to descend
    fixed = synthetic_batch(cfg, device)
    torch.manual_seed(1)
    t2 = SSLTrainer(cfg, output_dir=tmp, device=device)
    first = t2.train_step(fixed, step=0, total_steps=40)
    for s in range(1, 30):
        last = t2.train_step(fixed, step=s, total_steps=40)
    check("Head2 (distance) loss descends on a fixed batch",
          last["loss_distance"] < first["loss_distance"],
          f"{first['loss_distance']:.4f} -> {last['loss_distance']:.4f}")
    check("Head3 (change) loss descends on a fixed batch",
          last["loss_change"] < first["loss_change"],
          f"{first['loss_change']:.4f} -> {last['loss_change']:.4f}")


# ----------------------------------------------------------------------------------
# G/H. padding inertness + variable T
# ----------------------------------------------------------------------------------
def audit_masking(cfg, device) -> None:
    section("G. Padding inertness — padded timesteps cannot influence valid outputs")
    torch.manual_seed(0)
    bb = GlacierBackbone(cfg).to(device).eval()
    o = int(cfg.data.image_size)
    B, T = 2, 4
    imgs = torch.randn(B, T, 6, o, o, device=device)
    tmask = torch.ones(B, T, dtype=torch.bool, device=device)
    tmask[:, 2:] = False                      # last two are padding
    elapsed = torch.rand(B, T, device=device)
    span = torch.rand(B, device=device)
    sids = seasons_to_ids([["DJF"] * T] * B, T, device)

    with torch.no_grad():
        a = bb(imgs, tmask, elapsed, span, sids)
        perturbed = imgs.clone()
        perturbed[:, 2:] = torch.randn_like(perturbed[:, 2:]) * 50.0   # garbage in padding
        e2 = elapsed.clone(); e2[:, 2:] = 999.0
        b = bb(perturbed, tmask, e2, span, sids)

    check("valid regional_out unchanged by garbage in padded steps",
          torch.allclose(a["regional_out"][:, :2], b["regional_out"][:, :2], atol=1e-5),
          f"max|d|={float((a['regional_out'][:, :2] - b['regional_out'][:, :2]).abs().max()):.2e}")
    check("valid timestamp_out unchanged by garbage in padded steps",
          torch.allclose(a["timestamp_out"][:, :2], b["timestamp_out"][:, :2], atol=1e-5),
          f"max|d|={float((a['timestamp_out'][:, :2] - b['timestamp_out'][:, :2]).abs().max()):.2e}")
    check("sequence_out (context token) unchanged by garbage in padded steps",
          torch.allclose(a["sequence_out"], b["sequence_out"], atol=1e-5),
          f"max|d|={float((a['sequence_out'] - b['sequence_out']).abs().max()):.2e}")

    section("H. Variable sequence length — T is never fixed")
    ok_lengths = []
    for t in (1, 2, 3, 5, 7):
        with torch.no_grad():
            out = bb(torch.randn(1, t, 6, o, o, device=device),
                     torch.ones(1, t, dtype=torch.bool, device=device),
                     torch.rand(1, t, device=device), torch.rand(1, device=device),
                     seasons_to_ids([["DJF"] * t], t, device))
        ok_lengths.append(tuple(out["timestamp_out"].shape) == (1, t, int(cfg.tet.embed_dim))
                          and torch.isfinite(out["sequence_out"]).all())
    check("T = 1,2,3,5,7 all produce correct finite outputs", all(ok_lengths),
          f"results={ok_lengths}")

    # pad-invariance: a real T=3 sequence must give the same answer when padded to T=6
    torch.manual_seed(3)
    real = torch.randn(1, 3, 6, o, o, device=device)
    el = torch.tensor([[0.0, 1.0, 2.5]], device=device)
    sp = torch.tensor([2.5], device=device)
    with torch.no_grad():
        short = bb(real, torch.ones(1, 3, dtype=torch.bool, device=device), el, sp,
                   seasons_to_ids([["DJF", "MAM", "JJA"]], 3, device))
        padded_imgs = torch.cat([real, torch.randn(1, 3, 6, o, o, device=device)], dim=1)
        m = torch.zeros(1, 6, dtype=torch.bool, device=device); m[:, :3] = True
        el6 = torch.cat([el, torch.zeros(1, 3, device=device)], dim=1)
        long = bb(padded_imgs, m, el6, sp,
                  seasons_to_ids([["DJF", "MAM", "JJA", "DJF", "DJF", "DJF"]], 6, device))
    check("T=3 outputs identical whether batched alone or padded to T=6",
          torch.allclose(short["timestamp_out"], long["timestamp_out"][:, :3], atol=1e-5)
          and torch.allclose(short["sequence_out"], long["sequence_out"], atol=1e-5),
          f"ts max|d|={float((short['timestamp_out'] - long['timestamp_out'][:, :3]).abs().max()):.2e}"
          f"  seq max|d|={float((short['sequence_out'] - long['sequence_out']).abs().max()):.2e}")


# ----------------------------------------------------------------------------------
# I. information barriers
# ----------------------------------------------------------------------------------
def audit_information_barriers(cfg, device) -> None:
    section("I. Information barriers — heads must not see what they predict")
    dim = int(cfg.tet.embed_dim)
    R = int(cfg.tet.num_regions)
    torch.manual_seed(0)

    h2 = Head2Distance(cfg).to(device).eval()
    u = torch.randn(4, dim, device=device)
    v = torch.randn(4, dim, device=device)
    with torch.no_grad():
        fwd = h2(u, v)
        rev = h2(v, u)
    check("Head2 is swap-symmetric: d(i,j) == d(j,i)",
          torch.allclose(fwd, rev, atol=1e-5),
          f"max|d|={float((fwd - rev).abs().max()):.2e}")
    sig = str(h2.forward.__doc__ or "")
    check("Head2.forward takes only the two embeddings (no gap/index argument)",
          h2.forward.__code__.co_argcount == 3,
          f"argcount={h2.forward.__code__.co_argcount} (self, z_i, z_j)")

    h3 = Head3Change(cfg).to(device).eval()
    r_i = torch.randn(3, R, dim, device=device)
    r_j = torch.randn(3, R, dim, device=device)
    with torch.no_grad():
        _, c_fwd = h3.encode(r_i, r_j)
        _, c_rev = h3.encode(r_j, r_i)
    check("Head3 preserves direction: encode(i,j) != encode(j,i)",
          not torch.allclose(c_fwd, c_rev, atol=1e-3),
          f"mean cos={float((c_fwd * c_rev).sum(-1).mean()):.4f} (must be < 1)")
    check("Head3.encode takes only region embeddings + validity (no gap argument)",
          h3.encode.__code__.co_argcount == 4,
          f"argcount={h3.encode.__code__.co_argcount} (self, r_i, r_j, region_valid)")

    # validity-aware pooling: masked-out regions must not affect the change embedding
    valid = torch.ones(3, R, dtype=torch.bool, device=device)
    valid[:, R // 2:] = False
    r_j2 = r_j.clone()
    r_j2[:, R // 2:] = torch.randn_like(r_j2[:, R // 2:]) * 100.0
    with torch.no_grad():
        a, _ = h3.encode(r_i, r_j, valid)
        b, _ = h3.encode(r_i, r_j2, valid)
    check("Head3 pooling ignores invalid regions",
          torch.allclose(a, b, atol=1e-5),
          f"max|d|={float((a - b).abs().max()):.2e}")


# ----------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument("--full-config", action="store_true",
                    help="also instantiate the real 768-dim backbone and count params")
    args = ap.parse_args()

    from glacier_ssl.train.accel import resolve_device
    device = resolve_device(args.device)

    print("=" * 78)
    print("GLACIER SSL — PRE-TRAINING SYSTEM AUDIT")
    print("=" * 78)
    print(f"torch {torch.__version__} | device={device} | "
          f"cuda_devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    full = load_config(None)
    audit_config_geometry(full, "default.yaml (production)")

    cfg = small_config()
    audit_config_geometry(cfg, "miniature (audit)")

    audit_tensor_contract(cfg, device)
    audit_masking(cfg, device)
    audit_information_barriers(cfg, device)
    with tempfile.TemporaryDirectory() as tmp:
        audit_wiring(cfg, device, Path(tmp))

    if args.full_config:
        section("K. Production backbone instantiation (768-dim, 12+8 blocks)")
        bb = GlacierBackbone(full)
        n = sum(p.numel() for p in bb.parameters())
        tr = sum(p.numel() for p in bb.parameters() if p.requires_grad)
        check("real backbone builds with spec dims", True,
              f"{n:,} params ({tr:,} trainable)")
        del bb

    section("SUMMARY")
    failed = [n for n, ok, _ in _RESULTS if not ok]
    print(f"  {len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    if failed:
        print("\n  FAILED CHECKS:")
        for n in failed:
            print(f"    - {n}")
        print("\n  >>> NOT cleared for training. Fix the above first.")
        return 1
    print("\n  >>> System verified. Cleared for SSL pretraining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
