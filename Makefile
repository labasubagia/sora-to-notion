# Prefer virtualenv python at .venv/bin/python when available
ifneq ($(wildcard .venv/bin/python),)
PY := .venv/bin/python
else
PY ?= python
endif

IMAGE_CHATGPT = debug_chatgpt_images
IMAGE_SORA = debug_sora_images

.PHONY: help chatgpt_upload_to_notion chatgpt_upload_to_notion_remove \
	sora_cleanup_trash sora_cleanup_tasks \
	sora_upload_to_notion sora_upload_to_notion_trash sora_upload_to_notion_remove \
	clean-output-path

help:
	@echo "Available targets:"
	@echo "  chatgpt_upload_to_notion           Run chatgpt-upload-to-notion (no-remove)"
	@echo "  chatgpt_upload_to_notion_remove    Run chatgpt-upload-to-notion (remove)"
	@echo "  sora_cleanup_trash                 Run sora-cleanup-trash"
	@echo "  sora_cleanup_tasks                 Run sora-cleanup-tasks"
	@echo "  sora_upload_to_notion              Run sora-upload-to-notion (no-remove)"
	@echo "  sora_upload_to_notion_trash        Run sora-upload-to-notion (trash-in-sora)"
	@echo "  sora_upload_to_notion_remove       Run sora-upload-to-notion (remove-in-sora)"
	@echo "  clean-output-path                  Run clean-output-path (module)"

chatgpt_upload_to_notion:
	$(PY) main.py chatgpt-upload-to-notion --image-folder $(IMAGE_CHATGPT) --no-remove-in-chatgpt

chatgpt_upload_to_notion_remove:
	$(PY) main.py chatgpt-upload-to-notion --image-folder $(IMAGE_CHATGPT) --remove-in-chatgpt

sora_cleanup_trash:
	$(PY) main.py sora-cleanup-trash

sora_cleanup_tasks:
	$(PY) main.py sora-cleanup-tasks

sora_upload_to_notion:
	$(PY) main.py sora-upload-to-notion --image-folder $(IMAGE_SORA) --no-remove-in-sora

sora_upload_to_notion_trash:
	$(PY) main.py sora-upload-to-notion --image-folder $(IMAGE_SORA) --trash-in-sora

sora_upload_to_notion_remove:
	$(PY) main.py sora-upload-to-notion --image-folder $(IMAGE_SORA) --remove-in-sora

clean-output-path:
	$(PY) -m main clean-output-path
