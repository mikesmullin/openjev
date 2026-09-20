"""Real Laya predictions, with an explicit optional deterministic safety shield."""

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from .game import DIRECTIONS

DEFAULT_MODEL = "aac6fef/laya-multilingual-mlx"

# MLX-port checkpoint IDs and their upstream torch equivalents
# (repo, subfolder) for NandhaKishorM/laya style `laya.load(repo, subfolder=...)`.
TORCH_MODEL_MAP = {
    "aac6fef/laya-mlx": ("convaiinnovations/laya", None),
    "aac6fef/laya-multilingual-mlx": ("convaiinnovations/laya", "multilingual"),
    "aac6fef/laya-typed-decisions-mlx": ("convaiinnovations/laya", "typed-decisions"),
}


def resolve_backend(backend=None):
    """Pick 'mlx' or 'torch'. 'auto' prefers MLX on Apple Silicon, torch elsewhere."""
    value = (backend or "auto").lower()
    if value == "auto":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx.core  # noqa: F401

                return "mlx"
            except ImportError:
                pass
        return "torch"
    if value not in ("mlx", "torch"):
        raise ValueError("backend must be auto, mlx or torch")
    return value


def resolve_torch_model(model):
    """Map an MLX checkpoint ID (or anything else) to (repo_or_dir, subfolder)."""
    if model is None:
        repo, subfolder = TORCH_MODEL_MAP[DEFAULT_MODEL]
        return repo, subfolder
    if model in TORCH_MODEL_MAP:
        return TORCH_MODEL_MAP[model]
    return model, None


def load_torch_agent(model=None, device=None, upstream=None):
    """Load the upstream PyTorch Laya agent (CUDA/MPS/CPU). Returns (agent, repo, subfolder)."""
    import sys

    # Prefer the pinned .upstream checkout over any PyPI `laya` that may be installed.
    root = Path(upstream) if upstream else Path(__file__).resolve().parents[2] / ".upstream"
    if (root / "laya" / "agent.py").exists():
        sys.path.insert(0, str(root))
    try:
        import laya  # noqa: F401
    except ImportError:
        raise FileNotFoundError(
            "Torch backend needs the upstream checkout:\n"
            "  gh repo clone NandhaKishorM/laya .upstream\n"
            "  git -C .upstream checkout 6a5819129eb220570792e417e49723d697efd76f"
        )
    path = Path(str(model)).expanduser() if model else None
    if path is not None and path.is_dir():
        return laya.load(str(path), device=device), str(path), None
    repo, subfolder = resolve_torch_model(str(model) if model else None)
    if str(repo).startswith((".", "/", "~")):
        raise FileNotFoundError(f"Local checkpoint does not exist: {repo}")
    return laya.load(repo, device=device, subfolder=subfolder), repo, subfolder


def local_checkpoint(value=None):
    """Resolve a directory or an already cached Hub snapshot without network access."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if value is None:
        for path in ("models/hub/laya-multilingual-mlx", "models/laya-multilingual"):
            if Path(path).is_dir():
                return Path(path)
        value = DEFAULT_MODEL
    path = Path(value).expanduser()
    if path.is_dir():
        return path
    if str(value).startswith((".", "/", "~")):
        raise FileNotFoundError(f"Local checkpoint does not exist: {value}")
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(str(value), local_files_only=True))
    except Exception as error:
        raise FileNotFoundError(
            f"{value} is not cached. Download it before starting the offline demo:\n"
            f"  hf download {value} --local-dir models/snake\n"
            "  laya-snake --model models/snake"
        ) from error


def hardware_name():
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip().removeprefix("Apple ")
    return platform.machine()


def torch_metadata(repo, subfolder, agent, device_arg):
    try:
        import torch

        if agent.device.type == "cuda":
            hardware = torch.cuda.get_device_name(agent.device)
        else:
            hardware = hardware_name() + f" (torch-{agent.device.type})"
        versions = {}
        for name in ("torch", "transformers", "numpy", "rich", "tokenizers", "huggingface-hub"):
            try:
                versions[name] = version(name)
            except Exception:
                pass
    except Exception:
        hardware, versions = hardware_name(), {}
    dtype = str(getattr(agent, "dtype", ""))
    precision = {"torch.float32": "FP32", "torch.float16": "FP16", "torch.bfloat16": "BF16"}.get(
        dtype, dtype or "FP32"
    )
    return {
        "name": f"{repo}" + (f"/{subfolder}" if subfolder else ""),
        "backend": "torch",
        "precision": precision,
        "device": str(agent.device),
        "device_arg": device_arg or "auto",
        "hardware": hardware,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "network": "offline" if Path(str(repo)).exists() else "hub-cache",
        "policy": "Laya probabilities over planner features; optional cycle safety shield",
        "source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
    }


def checkpoint_metadata(path):
    meta = path / "mlx_config.json"
    values = json.loads(meta.read_text()) if meta.exists() else {}
    manifest = path / "manifest.json"
    source = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {
        "backend": "mlx",
        "precision": "FP16",
        "device": "Apple GPU",
        "name": values.get("repository", path.name),
        "source_revision": values.get("source_revision"),
        "weight_sha256_from_manifest": source.get("files", {})
        .get("model.safetensors", {})
        .get("sha256"),
        "hardware": hardware_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("mlx", "numpy", "rich", "tokenizers", "huggingface-hub")
        },
        "network": "offline",
        "policy": "Laya probabilities over planner features; optional cycle safety shield",
        "source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
    }


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    def __init__(
        self,
        model=None,
        *,
        guarded=True,
        prompt="compact",
        optimize=False,
        backend="auto",
        device=None,
        upstream=None,
    ):
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        self.backend = resolve_backend(backend)
        self.guarded = guarded
        self.prompt = prompt
        if self.backend == "mlx":
            from laya_mlx import Agent

            self.path = local_checkpoint(model)
            self.agent = Agent(
                self.path,
                dtype="float16",
                device="gpu",
                batch_size=3,
                compile=optimize,
                pad_to_multiple=16 if optimize else None,
                cache_prompts=optimize,
            )
            self.metadata = checkpoint_metadata(self.path)
            self.metadata["backend"] = "mlx"
            self.metadata["prompt"] = prompt
            self.metadata["optimization"] = (
                "compile + 16-token buckets + prefix cache" if optimize else "eager"
            )
        else:
            if optimize:
                print("Note: --optimize is an MLX-only path; ignoring on torch backend.")
            agent, repo, subfolder = load_torch_agent(model, device, upstream)
            self.agent = agent
            self.path = Path(str(repo)) if Path(str(repo)).exists() else repo
            self.metadata = torch_metadata(repo, subfolder, agent, device)
            self.metadata["prompt"] = prompt
            self.metadata["optimization"] = "eager (torch)"

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()
        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
            elif not move.safe:
                descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best progress toward food."
            else:
                descriptions[move.direction] = "Safe but less progress toward food."
        state = (
            f"Snake game. {len(safe)} safe directions available. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
            f"Open cells: {space}. Snake length: {len(game.body)}. "
            f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
                "criteria": descriptions,
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the snake?",
            },
            "food": {
                "type": "noul",
                "instructions": "Is food reachable through the currently empty cells?",
            },
        }
        if self.prompt == "compact":
            state = (
                f"Safe route: {'yes' if safe else 'no'}. "
                f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
            )
            questions["move"]["instructions"] = "Choose the best safe move toward food."
            questions["move"]["criteria"] = {
                m.direction: (
                    "Blocked. Collision."
                    if not m.legal
                    else "Unsafe. Traps the snake."
                    if not m.safe
                    else "Safe. Eat food now. Best."
                    if m.eats
                    else "Safe. Best route to food."
                    if m.direction == preferred
                    else "Safe. Slower route."
                )
                for m in moves
            }
            questions["risk"]["instructions"] = "Is a safe route available?"
            questions["food"]["instructions"] = "Is food reachable through empty cells?"
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = [*probabilities.values(), answers["risk"]["noul"], answers["food"]["noul"]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no move executed")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - answers["risk"]["noul"],
            food_reachable=answers["food"]["noul"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
            output_tokens=output["usage"].get("output_tokens", 0),
            safe_count=len(safe),
            planner_best=preferred,
        )
