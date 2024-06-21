.PHONY: quality style test docs

check_dirs := src tests examples docs scripts docker

# Check that source code meets quality standards

# this target runs checks on all files
quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)
	doc-builder style src/peft tests docs/source --max_len 119 --check_only

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	ruff check --fix $(check_dirs)
	ruff format $(check_dirs)
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

tests_examples_multi_gpu_bnb:
	python -m pytest -m "multi_gpu_tests and bitsandbytes" tests/test_gpu_examples.py $(if $(IS_GITHUB_CI),--report-log "multi_gpu_examples.log",)

tests_examples_single_gpu_bnb:
	python -m pytest -m "single_gpu_tests and bitsandbytes" tests/test_gpu_examples.py $(if $(IS_GITHUB_CI),--report-log "single_gpu_examples.log",)

tests_core_multi_gpu_bnb:
	python -m pytest -m "multi_gpu_tests and bitsandbytes" tests/test_common_gpu.py $(if $(IS_GITHUB_CI),--report-log "core_multi_gpu.log",)

tests_core_single_gpu_bnb:
	python -m pytest -m "single_gpu_tests and bitsandbytes" tests/test_common_gpu.py $(if $(IS_GITHUB_CI),--report-log "core_single_gpu.log",)

tests_gpu_bnb_regression:
	python -m pytest tests/bnb/test_bnb_regression.py $(if $(IS_GITHUB_CI),--report-log "bnb_regression_gpu.log",)

# For testing transformers tests for bnb runners
transformers_tests:
	RUN_SLOW=1 python -m pytest transformers-clone/tests/quantization/bnb $(if $(IS_GITHUB_CI),--report-log "transformers_tests.log",)

tests_regression:
	python -m pytest -s --regression tests/regression/ $(if $(IS_GITHUB_CI),--report-log "regression_tests.log",)

tests_torch_compile:
	python -m pytest tests/test_torch_compile.py $(if $(IS_GITHUB_CI),--report-log "compile_tests.log",)
