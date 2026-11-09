"""The Riva and Speech NIM compatibility frontend, a vLLM endpoint plugin."""

from vllm_riva_frontend.lifecycle import RivaFrontendPlugin
from vllm_riva_frontend.version import __version__

__all__ = ["RivaFrontendPlugin", "__version__", "plugin"]


# @spec ING-EP-001
def plugin() -> RivaFrontendPlugin:
    """Return the frontend plugin; the ``riva_frontend`` entry point."""
    return RivaFrontendPlugin()
