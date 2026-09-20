"""Laya typed decisions on Apple silicon with MLX.

On non-macOS hosts MLX is unavailable; the MLX runtime imports below degrade
to None and the torch backend (laya_mlx.snake.policy, upstream checkout in
.upstream) is the supported path.
"""

try:
    from .agent import Agent, RLAgent, load

    _MLX_AVAILABLE = True
except ImportError:  # MLX has no Linux wheels; snake torch backend stays usable
    Agent = None
    RLAgent = None
    load = None
    _MLX_AVAILABLE = False

try:
    from .router import DEFAULT_MODELS, RouteDecision, Router
except ImportError:
    DEFAULT_MODELS = {}
    RouteDecision = None
    Router = None

from .email import clean_email_body, email_state
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "detect_language",
    "detect_script",
    "is_english",
    "clean_email_body",
    "email_state",
    "email_questions",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
]
