"""Chart-shaped RFC-2 deployment rendering."""

import json
from pathlib import Path

import pytest
import yaml

from vllm_riva_frontend.deployment import (
    load_values,
    render_manifest,
)

ROOT = Path(__file__).parents[1]
VALUES_PATH = ROOT / "deploy" / "riva_frontend" / "home.values.yaml"
TEMPLATE_PATH = ROOT / "deploy" / "riva_frontend" / "manifest.template.yaml"


# @spec ENV-MOD-001, ENV-MOD-003, ENV-MOD-004
def test_production_module_renders_one_cohosted_server() -> None:
    values = load_values(VALUES_PATH)
    assert values["served_model_name"] != values["model"]
    rendered = render_manifest(
        values, TEMPLATE_PATH.read_text(encoding="utf-8")
    )
    documents = list(yaml.safe_load_all(rendered))
    assert [document["kind"] for document in documents] == [
        "ConfigMap",
        "Deployment",
        "Service",
    ]

    config = json.loads(documents[0]["data"]["riva_frontend.json"])
    assert config["deployment_image"] == values["image"]
    assert config["grpc_bind"] == "0.0.0.0:50051"
    assert config["session_idle_timeout"] == 60.0

    container = documents[1]["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]
    assert container["image"] == values["image"]
    assert container["command"] == ["/opt/venv-port/bin/vllm-omni"]
    assert args.count("--application-plugin") == 1
    assert args[args.index("--application-plugin") + 1] == "riva_frontend"
    assert args[args.index("--api-server-count") + 1] == "1"
    assert (
        args[args.index("--served-model-name") + 1]
        == values["served_model_name"]
    )
    assert "--omni" in args
    assert "--ws-max-size" in args
    assert "--h11-max-incomplete-event-size" in args
    assert "remote_provider" not in rendered
    assert "/v1/audio/translations" not in rendered


# @spec ENV-MOD-001, ENV-MOD-003
def test_deployment_values_fail_closed_before_render() -> None:
    values = load_values(VALUES_PATH)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    without_bound = dict(values)
    del without_bound["session_cleanup_timeout"]
    with pytest.raises(ValueError, match="missing deployment values"):
        render_manifest(without_bound, template)

    wrong_image = dict(values)
    wrong_image["deployment_image"] = "different"
    with pytest.raises(ValueError, match="must equal the pod image"):
        render_manifest(wrong_image, template)

    wrong_port = dict(values)
    wrong_port["grpc_service_port"] = 50052
    with pytest.raises(ValueError, match="must equal the port in grpc_bind"):
        render_manifest(wrong_port, template)


# @spec ENV-SEC-003
def test_frontend_values_cannot_acquire_a_security_profile() -> None:
    values = load_values(VALUES_PATH)
    values["grpc_tls_profile"] = "internal"
    with pytest.raises(ValueError, match="security is deployment-external"):
        render_manifest(values, TEMPLATE_PATH.read_text(encoding="utf-8"))
