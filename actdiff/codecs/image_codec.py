"""Keyframe image codecs: HiFiC (low / med / high) and MLIC++."""
import importlib
import os
import sys
import types
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from actdiff.paths import HIFIC_ROOT, MLICPP_ROOT


@contextmanager
def Repo_On_Path(root):
    root = str(root)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if added and root in sys.path:
            sys.path.remove(root)


@contextmanager
def Bind_Namespace(name, path):
    """Temporarily map top-level package `name` to `path` (MLIC++ uses generic names like `utils`)."""
    prev = sys.modules.get(name)
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg
    try:
        yield
    finally:
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


def Load_Mlic(device, checkpoint_path):
    import compressai.ops

    if not hasattr(compressai.ops, "ste_round"):  # renamed to quantize_ste in newer CompressAI
        compressai.ops.ste_round = compressai.ops.quantize_ste
    with Repo_On_Path(MLICPP_ROOT), Bind_Namespace("models", MLICPP_ROOT / "models"), Bind_Namespace(
        "utils", MLICPP_ROOT / "utils"
    ), Bind_Namespace("config", MLICPP_ROOT / "config"):
        model_cls = importlib.import_module("models.mlicpp").MLICPlusPlus
        model_config = importlib.import_module("config.config").model_config
        testing = importlib.import_module("utils.testing")

    model = model_cls(config=model_config()).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    return model.eval(), testing.compress_one_image, testing.decompress_one_image


def Patch_Hific_Dependencies():
    """HiFiC targets old NumPy / scikit-image; restore the aliases it imports."""
    for name, typ in {"int": int, "float": float, "bool": bool, "object": object, "complex": complex}.items():
        if not hasattr(np, name):
            setattr(np, name, typ)
    import skimage.measure

    if not hasattr(skimage.measure, "compare_ssim"):
        from skimage.metrics import structural_similarity

        skimage.measure.compare_ssim = structural_similarity


def Load_Hific(device, checkpoint_path, work_dir):
    Patch_Hific_Dependencies()
    with Repo_On_Path(HIFIC_ROOT):
        prepare_model = importlib.import_module("compress").prepare_model
        hific_utils = importlib.import_module("src.compression.compression_utils")
    os.makedirs(work_dir, exist_ok=True)
    model, loaded_args = prepare_model(checkpoint_path, work_dir)
    return model.to(device).eval(), loaded_args, hific_utils


def Pad_To_64(image):
    _, _, H, W = image.shape
    padded = F.pad(image, (0, (64 - W % 64) % 64, 0, (64 - H % 64) % 64), mode="constant", value=0)
    return padded, H, W


class KeyframeCodec:
    """Compresses [1, 3, H, W] images in [0, 1] to files, and back."""

    def __init__(self, args):
        self.name = args.keyframe_codec
        self.device = args.device
        if self.name == "mlic":
            self.model, self.mlic_compress, self.mlic_decompress = Load_Mlic(
                self.device, args.mlic_checkpoint
            )
        else:
            checkpoint = {
                "hific-low": args.hific_low_checkpoint,
                "hific-med": args.hific_med_checkpoint,
                "hific-high": args.hific_high_checkpoint,
            }[self.name]
            self.model, hific_args, self.hific_utils = Load_Hific(self.device, checkpoint, args.output_dir)
            self.normalize_input = bool(getattr(hific_args, "normalize_input_image", True))

    @torch.no_grad()
    def Compress(self, image, stream_dir, name):
        os.makedirs(stream_dir, exist_ok=True)
        if self.name == "mlic":
            padded, H, W = Pad_To_64(image)
            self.mlic_compress(self.model, padded, stream_dir, H, W, name)
            return
        x = image.to(self.device, dtype=torch.float32)
        if self.normalize_input:
            x = x * 2.0 - 1.0
        self.hific_utils.save_compressed_format(
            self.model.compress(x), out_path=os.path.join(stream_dir, name)
        )

    @torch.no_grad()
    def Decompress(self, stream_dir, name):
        """Returns the decoded keyframe as [3, H, W] in [0, 1]."""
        if self.name == "mlic":
            x_hat, _ = self.mlic_decompress(self.model, stream_dir, name)
            return x_hat[0]
        x_hat = self.model.decompress(self.hific_utils.load_compressed_format(os.path.join(stream_dir, name)))
        if float(x_hat.min()) < -0.1:  # some checkpoints decode to [-1, 1]
            x_hat = (x_hat + 1.0) / 2.0
        return x_hat.clamp(0.0, 1.0)[0]
