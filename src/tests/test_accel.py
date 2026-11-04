"""Tests for GPU-acceleration helpers (device resolution, AMP gating, DataParallel policy,
autocast runner). CUDA paths can't execute on a CPU box, but the gating logic and the
no-op CPU behavior are verified here so the trainer stays correct on both."""
import torch

from _helpers import small_config
from glacier_ssl.backbone.backbone import GlacierBackbone
from glacier_ssl.train.accel import (resolve_device, use_data_parallel, amp_settings,
                                     AutocastRunner, autocast_ctx)


def test_resolve_device_falls_back_to_cpu_without_cuda():
    dev = resolve_device("cuda")
    if not torch.cuda.is_available():
        assert dev.type == "cpu"           # graceful fallback, no crash
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")


def test_data_parallel_disabled_on_cpu():
    assert use_data_parallel("auto", torch.device("cpu")) is False
    assert use_data_parallel(True, torch.device("cpu")) is False


def test_amp_disabled_on_cpu():
    enabled, dtype = amp_settings(True, "float16", torch.device("cpu"))
    assert enabled is False                 # AMP only on CUDA
    enabled_bf, dtype_bf = amp_settings(True, "bfloat16", torch.device("cpu"))
    assert dtype_bf == torch.bfloat16


def test_autocast_runner_is_transparent_when_disabled():
    cfg = small_config()
    bb = GlacierBackbone(cfg).eval()
    runner = AutocastRunner(bb, amp_enabled=False, amp_dtype=torch.float16,
                            device_type="cpu")
    imgs = torch.randn(1, 2, 6, 128, 128)
    m = torch.ones(1, 2, dtype=torch.bool)
    e = torch.rand(1, 2); sp = torch.rand(1)
    season_ids = torch.randint(0, 4, (1, 2))
    with torch.no_grad():
        a = bb(imgs, m, e, sp, season_ids)
        b = runner(imgs, m, e, sp, season_ids)
    assert torch.allclose(a["sequence_out"], b["sequence_out"], atol=1e-6)


def test_autocast_ctx_noop_when_disabled():
    import contextlib
    ctx = autocast_ctx("cpu", torch.float16, enabled=False)
    assert isinstance(ctx, contextlib.nullcontext)


def test_backbone_accepts_season_id_tensor_or_list():
    """DataParallel needs a season-id TENSOR; direct callers may pass the list form."""
    cfg = small_config()
    bb = GlacierBackbone(cfg).eval()
    imgs = torch.randn(1, 2, 6, 128, 128)
    m = torch.ones(1, 2, dtype=torch.bool)
    e = torch.rand(1, 2); sp = torch.rand(1)
    with torch.no_grad():
        out_list = bb(imgs, m, e, sp, [["DJF", "MAM"]])
        out_tensor = bb(imgs, m, e, sp, torch.tensor([[0, 1]]))
    assert torch.allclose(out_list["sequence_out"], out_tensor["sequence_out"], atol=1e-6)
