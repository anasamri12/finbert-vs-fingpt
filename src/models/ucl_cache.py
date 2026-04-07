from __future__ import annotations

import getpass
import os
from pathlib import Path


def configure_ucl_scratch_cache() -> None:
    """Prefer /scratch0 for large runtime caches on the UCL servers.

    This keeps big Hugging Face model downloads, Triton compilation artifacts,
    KaggleHub cache files, and temporary files off the small home-directory
    quota. Existing explicit environment settings always win.
    """

    if os.environ.get("CODEX_UCL_SCRATCH_CACHE_CONFIGURED") == "1":
        return

    username = os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser()
    scratch_user_root = Path("/scratch0") / username
    if not scratch_user_root.exists():
        return

    cache_root = scratch_user_root / "fyp_cache"
    xdg_root = cache_root / "xdg"
    hf_root = cache_root / "huggingface"
    triton_root = cache_root / "triton"
    torch_root = cache_root / "torch"
    kaggle_root = cache_root / "kagglehub"
    tmp_root = cache_root / "tmp"

    for path in (
        cache_root,
        xdg_root,
        hf_root,
        hf_root / "hub",
        hf_root / "transformers",
        hf_root / "datasets",
        hf_root / "assets",
        triton_root,
        torch_root,
        kaggle_root,
        tmp_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_root))
    os.environ.setdefault("HF_HOME", str(hf_root))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_root / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_root / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_root / "datasets"))
    os.environ.setdefault("HF_ASSETS_CACHE", str(hf_root / "assets"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_root))
    os.environ.setdefault("TORCH_HOME", str(torch_root))
    os.environ.setdefault("KAGGLEHUB_CACHE", str(kaggle_root))
    os.environ.setdefault("TMPDIR", str(tmp_root))
    os.environ.setdefault("TMP", str(tmp_root))
    os.environ.setdefault("TEMP", str(tmp_root))
    os.environ["CODEX_UCL_SCRATCH_CACHE_CONFIGURED"] = "1"

    print(f"[cache] Using /scratch0 cache root: {cache_root}")
