# Makefile for running MetaMathQA experiments.

# --- Configuration ---
PYTHON := python
RUN_SCRIPT := run.py
EXPERIMENTS_DIR := experiments
RESULTS_DIR := results

# --- Automatic Experiment and Result Discovery ---

# 1. Find all experiment directories by looking for adapter_config.json files.
#    This gives us a list like: experiments/lora/llama-3.2-3B-rank32 ...
EXPERIMENT_PATHS := $(patsubst %/adapter_config.json,%,$(shell find $(EXPERIMENTS_DIR) -name "adapter_config.json"))

# 2. Define a function to replace all occurrences of a character in a string.
#    This is needed to replicate the result naming logic from run.py (e.g., "lora/foo" -> "lora-foo").
#    Usage: $(call replace-all, string, char_to_replace, replacement_char)
replace-all = $(if $(findstring $(2),$(1)),$(call replace-all,$(subst $(2),$(3),$(1)),$(2),$(3)),$(1))

# 3. Define a function to convert an experiment path to its flat result file path.
#    e.g., "experiments/lora/llama-3.2-3B-rank32" -> "results/lora-llama-3.2-3B-rank32.json"
exp_to_res = $(RESULTS_DIR)/$(call replace-all,$(patsubst $(EXPERIMENTS_DIR)/%,%,$(1)),/,--).json

# 4. Generate the list of all target result files we want to build.
RESULT_FILES := $(foreach exp,$(EXPERIMENT_PATHS),$(call exp_to_res,$(exp)))


# --- Main Rules ---

# The default 'all' target depends on all possible result files.
# Running `make` or `make all` will check and run any outdated or missing experiments.
all: $(RESULT_FILES)


# --- Dynamic Rule Generation ---

# This is the core logic. We dynamically generate a specific Makefile rule for each experiment found.
# This avoids a complex pattern rule and makes the logic clearer.
define EXPERIMENT_template
# Input $1: The full experiment path (e.g., experiments/lora/llama-3.2-3B-rank32)

# Define the rule:
# The target is the result file (e.g., results/lora-llama-3.2-3B-rank32.json).
# The dependencies are its config files, code changes need to be audited manually since they can
# vary in degree of importance. Note that we explicitly ignore when the script fails to run
# so that the other experiments still have a chance to run.
$(call exp_to_res,$(1)): $(1)/adapter_config.json $(wildcard $(1)/training_params.json)
	@echo "---"
	@echo "Running experiment: $(1)"
	-$(PYTHON) $(RUN_SCRIPT) -v $(1)
	@echo "Finished: $$@"
	@echo "---"

endef

# This command iterates through every found experiment path and evaluates the template,
# effectively stamping out a unique, explicit rule for each one.
$(foreach exp_path,$(EXPERIMENT_PATHS),$(eval $(call EXPERIMENT_template,$(exp_path))))


# --- Utility Rules ---

.PHONY: all clean list dump_rules

# The 'clean' rule removes all generated results.
clean:
	@echo "Cleaning results directory..."
	@([ -n "$(wildcard $(RESULTS_DIR)/*.json)" ] && rm $(RESULTS_DIR)/*.json) || exit 0

# The 'list' rule is for debugging. It shows the discovered experiments
# and the result files the Makefile expects to create for them.
list:
	@echo "Discovered experiment configurations:"
	@$(foreach exp,$(EXPERIMENT_PATHS),echo "  - $(exp)/adapter_config.json";)
	@echo "\nTarget result files:"
	@$(foreach res,$(RESULT_FILES),echo "  - $(res)";)

# The 'dump_rules' rule is for debugging. It dumps all dynamically defined rules.
define newline


endef
define DUMPED_RULES
	$(foreach exp_path,$(EXPERIMENT_PATHS),$(call EXPERIMENT_template,$(exp_path)))
endef

dump_rules:
	@echo -e "$(subst $(newline),\n,${DUMPED_RULES})"
