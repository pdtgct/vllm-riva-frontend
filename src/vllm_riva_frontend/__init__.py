"""Riva and Speech NIM endpoint plugin for vLLM core."""

from vllm_riva_frontend.lifecycle import (
    PluginContext,
    PluginLifetime,
    RivaFrontendPlugin,
)

__all__ = [
    "PluginContext",
    "PluginLifetime",
    "RivaFrontendPlugin",
    "plugin",
]

#: Entry-point object. vLLM loads the class and calls it with no arguments.
plugin = RivaFrontendPlugin
