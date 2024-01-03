<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Configuration

[`PeftConfigMixin`] is the base configuration class for storing the adapter configuration of a [`PeftModel`], and [`PromptLearningConfig`] is the base configuration class for soft prompt methods (p-tuning, prefix tuning, and prompt tuning). These base classes contain methods for saving and loading model configurations from the Hub, specifying the PEFT method to use, type of task to perform, and model configurations like number of layers and number of attention heads.

## PeftConfigMixin

[[autodoc]] config.PeftConfigMixin
    - all

## PeftConfig

[[autodoc]] PeftConfig
    - all

## PromptLearningConfig

[[autodoc]] PromptLearningConfig
    - all
