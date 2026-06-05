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

# PEFT

🤗 PEFT (Parameter-Efficient Fine-Tuning) is a library for efficiently adapting large pretrained models to various downstream applications without fine-tuning all of a model's parameters because it is prohibitively costly. PEFT methods only fine-tune a small number of (extra) model parameters - significantly decreasing computational and storage costs - while yielding performance comparable to a fully fine-tuned model. This makes it more accessible to train and store large language models (LLMs) and other big models on consumer hardware.

PEFT is integrated with the Transformers, Diffusers, and Accelerate libraries to provide a faster and easier way to load, train, and use large models for inference.

<div class="flex justify-center">
  <div class="flex flex-col basis-1/4">
    There are numerous methods to "adapt" existing models, often extensively integrating into the model. PEFT can be thought of as a framework for arbitrary methods of model adaption (modifying weights, wrapping layers, manipulating KV-caches, ...) while also serving as a reference implementation for many fine-tuning methods.
  </div>
  <div class="flex flex-col basis-3/4 pl-10 pr-10"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft//adapter_installation.png" width="100%"></div>
</div>

<div class="mt-10">
  <div class="w-full flex flex-col space-y-2 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="quicktour"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Quicktour</div>
      <p class="text-gray-700">Start here if you're new to 🤗 PEFT to get an overview of the library's main features, and how to train a model with a PEFT method.</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./methods/overview"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Method overview</div>
      <p class="text-gray-700">Learn about the different categories of PEFT methods to get an orientation what to use with your model.</p>
    </a>
  </div>
</div>

