.PHONY: quality style test docs

check_dirs := src tests examples docs

# Check that source code meets quality standards

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	ruff $(check_dirs)
	doc-builder style src/peft tests docs/source --max_len 119 --check_only

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	black $(check_dirs)
	ruff $(check_dirs) --fix
	doc-builder style src/peft tests docs/source --max_len 119

test:
	python -m pytest -n 3 tests/ $(if $(IS_GITHUB_CI),--report-log "ci_tests.log",)

tests_examples_multi_gpu:
	python -m pytest -m multi_gpu_tests tests/test_gpu_examples.py $(if $(IS_GITHUB_CI),--report-log "multi_gpu_examples.log",)

tests_examples_single_gpu:
	python -m pytest -m single_gpu_tests tests/test_gpu_examples.py $(if $(IS_GITHUB_CI),--report-log "single_gpu_examples.log",)

tests_core_multi_gpu:
	python -m pytest -m multi_gpu_tests tests/test_common_gpu.py $(if $(IS_GITHUB_CI),--report-log "core_multi_gpu.log",)

tests_core_single_gpu:
	python -m pytest -m single_gpu_tests tests/test_common_gpu.py $(if $(IS_GITHUB_CI),--report-log "core_single_gpu.log",)

tests_common_gpu:
	python -m pytest tests/test_decoder_models.py $(if $(IS_GITHUB_CI),--report-log "common_decoder.log",)
	python -m pytest tests/test_encoder_decoder_models.py $(if $(IS_GITHUB_CI),--report-log "common_encoder_decoder.log",)
