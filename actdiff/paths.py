"""Repository layout: third-party submodules and default checkpoint locations."""
import sys
from pathlib import Path

from actdiff.utils.scoped_imports import Install_Scoped_Imports

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

DIFFUSION_AS_SHADER_ROOT = THIRD_PARTY_DIR / "DiffusionAsShader"
ALLTRACKER_ROOT = THIRD_PARTY_DIR / "alltracker"
MLICPP_ROOT = THIRD_PARTY_DIR / "MLIC" / "MLIC++"
HIFIC_ROOT = THIRD_PARTY_DIR / "HiFiC"

# Repos whose bare top-level imports (e.g. `import utils`) must resolve inside the repo itself.
SCOPED_REPOS = {
    "alltracker": ALLTRACKER_ROOT,
    "DiffusionAsShader": DIFFUSION_AS_SHADER_ROOT,
    "MLIC++": MLICPP_ROOT,
    "HiFiC": HIFIC_ROOT,
}

_imports_ready = False


def Setup_Third_Party_Imports():
    """Make the submodules importable (e.g. `from alltracker.nets ...`) without name clashes."""
    global _imports_ready
    if _imports_ready:
        return
    missing = [f"{name}: {root}" for name, root in SCOPED_REPOS.items() if not root.exists()]
    if missing:
        raise FileNotFoundError(
            "Third-party submodules not found:\n  "
            + "\n  ".join(missing)
            + "\nRun `git submodule update --init --recursive` (or `pixi run import`)."
        )
    Install_Scoped_Imports(SCOPED_REPOS)
    sys.path.insert(0, str(THIRD_PARTY_DIR))
    _imports_ready = True
