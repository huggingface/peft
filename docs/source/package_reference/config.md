<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Configuration

The configuration classes stores the configuration of a [`PeftModel`], PEFT adapter models, and the configurations of [`PrefixTuning`], [`PromptTuning`], and [`PromptEncoder`]. They contain methods for saving and loading model configurations from the Hub, specifying the PEFT method to use, type of task to perform, and model configurations like number of layers and number of attention heads.

## PeftConfigMixin

[[autodoc]] config.PeftConfigMixin
    - all

## PeftConfig

[[autodoc]] PeftConfig
    - all

## PromptLearningConfig

[[autodoc]] PromptLearningConfig
    - all
