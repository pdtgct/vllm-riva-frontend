"""Render the chart-shaped ``riva_frontend`` deployment module."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from string import Template
from typing import Any

import yaml

from vllm_riva_frontend.config import FrontendConfig, load_plugin_config

_PLUGIN_FIELDS = frozenset(field.name for field in fields(FrontendConfig))
_METADATA_FIELDS = frozenset({"deployment_image", "pin", "precision_policy"})
_DEPLOYMENT_FIELDS = frozenset(
    {
        "service_name",
        "image",
        "image_pull_policy",
        "gpu_class",
        "gpu_resource",
        "workspace_claim_name",
        "model",
        "served_model_name",
        "stage_config_path",
        "http_host",
        "http_port",
        "grpc_service_port",
    }
)
_REQUIRED_FIELDS = (_PLUGIN_FIELDS | _METADATA_FIELDS | _DEPLOYMENT_FIELDS) - {
    "grpc_keepalive_seconds"
}
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_FORBIDDEN_SECURITY_STEMS = (
    "tls",
    "ssl",
    "api_key",
    "credential",
    "auth",
    "encryption",
)


def load_values(path: Path) -> dict[str, Any]:
    """Load one scalar YAML mapping without coercing values to strings."""
    try:
        decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read deployment values: {path}") from error
    if not isinstance(decoded, dict):
        raise ValueError("deployment values must be one YAML mapping")
    values: dict[str, Any] = {}
    for key, value in decoded.items():
        if not isinstance(key, str):
            raise ValueError("deployment value names must be strings")
        if isinstance(value, (dict, list)):
            raise ValueError(
                f"deployment value {key!r} must be one scalar value"
            )
        values[key] = value
    return values


def _require_deployment_values(values: dict[str, Any]) -> None:
    """Validate module structure before rendering any manifest."""
    forbidden = sorted(
        name
        for name in values
        if any(stem in name.lower() for stem in _FORBIDDEN_SECURITY_STEMS)
    )
    if forbidden:
        raise ValueError(
            "transport security is deployment-external, not a frontend "
            f"value: {', '.join(forbidden)}"
        )
    unknown = set(values) - _REQUIRED_FIELDS
    missing = _REQUIRED_FIELDS - set(values)
    if unknown:
        raise ValueError(
            f"unknown deployment values: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ValueError(
            f"missing deployment values: {', '.join(sorted(missing))}"
        )
    service_name = values["service_name"]
    if (
        not isinstance(service_name, str)
        or len(service_name) > 63
        or _DNS_LABEL.fullmatch(service_name) is None
    ):
        raise ValueError("service_name must be one Kubernetes DNS label")
    for name in (
        "image",
        "image_pull_policy",
        "gpu_class",
        "gpu_resource",
        "workspace_claim_name",
        "model",
        "served_model_name",
        "stage_config_path",
        "http_host",
    ):
        value = values[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if values["image_pull_policy"] not in {"Always", "IfNotPresent", "Never"}:
        raise ValueError("image_pull_policy is not a Kubernetes pull policy")
    for name in ("http_port", "grpc_service_port"):
        value = values[name]
        if type(value) is not int or not 1 <= value <= 65535:
            raise ValueError(f"{name} must be an integer in 1..65535")
    if values["deployment_image"] != values["image"]:
        raise ValueError("deployment_image must equal the pod image")


def render_manifest(values: dict[str, Any], template_text: str) -> str:
    """Validate values and render one deployment manifest.

    The embedded plugin configuration is generated from the same values
    mapping and validated through the runtime loader before rendering.
    """
    _require_deployment_values(values)
    raw_plugin = {
        name: values[name]
        for name in _PLUGIN_FIELDS | _METADATA_FIELDS
        if name in values
    }
    config, metadata = load_plugin_config(json.dumps(raw_plugin))

    bind_port = int(config.grpc_bind.rpartition(":")[2])
    if bind_port != values["grpc_service_port"]:
        raise ValueError("grpc_service_port must equal the port in grpc_bind")
    resolved_plugin = {
        **{
            name: value
            for name, value in vars(config).items()
            if value is not None
        },
        "deployment_image": metadata.image,
        "pin": metadata.pin,
        "precision_policy": metadata.precision_policy,
    }
    plugin_json = json.dumps(
        resolved_plugin,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    substitutions = {name.upper(): str(value) for name, value in values.items()}
    substitutions["PLUGIN_CONFIG_JSON"] = "\n".join(
        f"    {line}" for line in plugin_json.splitlines()
    )
    try:
        return Template(template_text).substitute(substitutions)
    except KeyError as error:
        raise ValueError(
            f"manifest template references unknown value: {error.args[0]}"
        ) from error


def render_files(
    *, values_path: Path, template_path: Path, output_path: Path
) -> None:
    """Render one values/template pair atomically into its output path."""
    values = load_values(values_path)
    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(
            f"cannot read deployment template: {template_path}"
        ) from error
    rendered = render_manifest(values, template_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    """Render a deployment module from its one values file."""
    parser = argparse.ArgumentParser(prog="vllm-riva-frontend-render")
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        render_files(
            values_path=args.values,
            template_path=args.template,
            output_path=args.output,
        )
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
