"""Downstream Riva and Speech NIM frontend for vLLM-Omni."""

from vllm_riva_frontend.lifecycle import PluginContext, PluginLifetime

__all__ = ["PluginContext", "PluginLifetime", "plugin"]


# @spec ING-VEH-001, ING-VEH-009, ING-VEH-011
class _Plugin:
    """Typed callable entry point with explicit configuration cardinality."""

    config_optional = False

    def __call__(self, context: PluginContext) -> PluginLifetime:
        """Return the explicitly selected plugin's host-bound lifetime."""
        return PluginLifetime(context)


plugin = _Plugin()
