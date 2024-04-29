<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tuners

A tuner (or adapter) is a module that can be plugged into a `torch.nn.Module`. [`BaseTuner`] base class for other tuners and provides shared methods and attributes for preparing an adapter configuration and replacing a target module with the adapter module. [`BaseTunerLayer`] is a base class for adapter layers. It offers methods and attributes for managing adapters such as activating and disabling adapters.

## BaseTuner

[[autodoc]] tuners.tuners_utils.BaseTuner

## BaseTunerLayer

[[autodoc]] tuners.tuners_utils.BaseTunerLayer