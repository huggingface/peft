.PHONY: quality style test docs

check_dirs := src tests examples

# Check that source code meets quality standards

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	doc-builder style src tests --max_len 119 --check_only

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	black $(check_dirs)
	isort $(check_dirs)
	doc-builder style src tests --max_len 119
	