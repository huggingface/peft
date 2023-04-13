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
	pytest -n 3 tests/

tests_examples_multi_gpu:
	pytest -m multi_gpu_tests tests/test_gpu_examples.py

tests_examples_single_gpu:
	pytest -m single_gpu_tests tests/test_gpu_examples.py

tests_core_multi_gpu:
	pytest -m multi_gpu_tests tests/test_common_gpu.py

tests_core_single_gpu:
	pytest -m single_gpu_tests tests/test_common_gpu.py

tests_common_gpu:
	pytest tests/test_decoder_models.py 
	pytest tests/test_encoder_decoder_models.py 