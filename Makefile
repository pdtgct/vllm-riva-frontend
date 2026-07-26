.PHONY: riva-frontend-render

PYTHON ?= .venv/bin/python
RIVA_FRONTEND_VALUES ?= deploy/riva_frontend/home.values.yaml
RIVA_FRONTEND_TEMPLATE ?= deploy/riva_frontend/manifest.template.yaml
RIVA_FRONTEND_MANIFEST ?= deploy/riva_frontend/rendered.yaml

riva-frontend-render:
	$(PYTHON) -m vllm_riva_frontend.deployment \
		--values $(RIVA_FRONTEND_VALUES) \
		--template $(RIVA_FRONTEND_TEMPLATE) \
		--output $(RIVA_FRONTEND_MANIFEST)
