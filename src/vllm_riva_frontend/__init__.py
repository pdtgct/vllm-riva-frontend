"""Downstream Riva and Speech NIM frontend for vLLM-Omni."""

from vllm_riva_frontend.lifecycle import PluginContext, PluginLifetime

__all__ = ["PluginContext", "PluginLifetime", "plugin"]


# @spec ING-VEH-001, ING-VEH-009, ING-VEH-011
class _Plugin:
    """Typed callable entry point with explicit configuration cardinality."""

    #: D5: the host reads this static attribute to decide whether a
    #: missing ``--application-plugin-config riva_frontend=...`` is an
    #: install-time error.  Absent configuration resolves the qualified
    #: zero-config default profile (see ``config.load_plugin_config``),
    #: so it is never required.
    config_optional = True

    def __call__(self, context: PluginContext) -> PluginLifetime:
        """Return the explicitly selected plugin's host-bound lifetime."""
        return PluginLifetime(context)


plugin = _Plugin()
