# Comparison of PEFT Methods

The goal of this project is to provide replicable experiments that produce outcomes allowing us to compare different PEFT methods with one another. This gives you more information to make an informed decision about which methods best fit your use case and what trade-offs to expect.

## Community Contributions

We envision the PEFT method comparison project as an ongoing endeavor with heavy involvement from the community. As maintainers, it is impossible for us to know all the perfect hyperparameters for each method or to predict all the use cases that PEFT users may have. As a consequence, community contributions are very welcome.

Below, we outline all the ways you can contribute to this project.

### Creating New Experiments

Creating a new experiment requires setting up a new PEFT configuration for us to test. This will result in one more data point being added to the total comparison.

Working on this is especially relevant if:

1. You are the author of a paper whose method is introduced in PEFT, or worked on the PEFT integration, and know what hyperparameters work best.
2. You have experience with a specific method and want to share your knowledge with the community.

Of course, you can contribute even without meeting these criteria. Please follow the instructions below.

#### How to Add New Experiments

Start by navigating to one of the existing experiment folders, e.g. `peft/method_comparison/MetaMathQA`, if your experiment involves using the [MetaMathQA dataset](https://huggingface.co/datasets/meta-math/MetaMathQA). There, create a new directory inside the `experiments/<method-name>` folder using a descriptive name. For example, if you want to test LoRA with rank 123 using Llama-3.2 3B as the base model, you could name the folder `experiments/lora/llama-3.2-3B-rank123`.

Inside this directory, you will find a default configuration file called `default_training_params.json`, which contains the default parameters used in the `run.py` training script. Create a new JSON file containing all the parameters you want to modify compared to the defaults, and save it as `training_params.json` in the newly created folder. If you are satisfied with all the default training parameters, you can skip this step.

Finally, you need to create a PEFT configuration file for the PEFT method you want to add. This should be a JSON file called `adapter_config.json`, placed in the same directory. Below is an example of how this could look:

```python
from peft import LoraConfig
config = LoraConfig(r=123)
config.save_pretrained("experiments/lora/llama-3.2-3B-rank123/")
```

Once you've created the configuration files for your experiment, please [create a PR on PEFT](https://github.com/huggingface/peft/pulls). After it is reviewed and merged, we will run it on our hardware to ensure that the results are comparable. Of course, it is best if you run the experiment at least once on your hardware to verify that the proposed settings work well.

#### Considerations When Adding New Experiments

When adding a new experiment, please consider the following points:

1. Avoid changing too many training parameters at once, as this would make it difficult to compare results with existing ones. For example, if all existing results were created with 5000 training steps but your result uses 10000 steps, it would be unclear whether an improvement in the test score is due to the PEFT method itself or simply due to longer training. Similarly, using a completely different base model, especially if it is significantly more capable, does not contribute to a fair comparison.
2. Avoid suggesting configurations that are very close to existing ones. For example, if there is already an experiment with LoRA and rank 123, do not add an experiment with LoRA and rank 124.
3. Experiments for less-tested methods are more valuable than additional experiments for widely tested methods.
4. Do not edit existing experiments, always create new ones.
5. If you found hyper parameters that work especially well with a given method but are not trivial to find out, consider updating the PEFT documentation of that method so that other users can benefit from your findings.

### Updating the Training Script

We provide a training script that includes features typically useful for improving training outcomes, such as AMP support, a cosine learning rate schedule, etc. However, there is always room for improvement. For example, at the time of writing, the script does not support gradient accumulation. Therefore, PRs that extend the training script are welcome.

#### How to Update the Training Script

Follow the same process as when contributing to PEFT in general (see the [contribution guidelines](https://huggingface.co/docs/peft/developer_guides/contributing)). If the same training script is used across multiple datasets, please ensure that all relevant scripts are updated accordingly.

#### Considerations When Updating the Training Script

1. Updates should be backward-compatible. By default, any new features should be disabled to ensure that existing results remain valid. For example, if you add gradient accumulation, ensure it is disabled by default so that new experiments must opt in.
2. Before adding a bug fix that could invalidate existing results, consider whether the trade-off is worthwhile. If we already have many experimental results, rerunning all of them can be expensive. If the bug fix is not critical, it may not be worth invalidating previous results. However, if you discover a significant bug that could meaningfully impact outcomes, it should be addressed.
3. Avoid unnecessary complexity. While we could add support for DeepSpeed, FSDP, etc., doing so would add significant complexity, exclude users with limited hardware, and is unlikely to alter the relative performance of different PEFT methods.
4. Minimize reliance on specific training frameworks. For example, we deliberately avoid using the `Trainer` class from transformers or PyTorch Lightning. This ensures transparency, making it easier to understand the training process and replicate results over time. If a training framework were used, we would have to pin the version or risk future incompatibilities.

### Adding a New Dataset

Adding a new dataset increases the breadth and usefulness of the PEFT method comparison. The goal is not necessarily to outperform benchmarks or replicate paper results, but to fairly compare different PEFT methods in a way that is useful for PEFT users. If this involves replicating an experiment from a paper, that is great, but it is not a requirement.

#### How to Add a New Dataset

The easiest way to add support for a new dataset is to copy an existing setup, such as `method_comparison/MetaMathQA`, rename it, and modify `data.py`, as well as any other necessary parts of the code. Ideally, as much existing code as possible should be reused. The general folder structure and experiment logging format should remain consistent.

After adding the dataset, ensure it functions correctly and produces meaningful results by running at least one experimental setup, such as using LoRA with default settings.

#### Considerations When Adding a New Dataset

1. Before beginning, it is best to open an [issue on PEFT](https://github.com/huggingface/peft/issues) to share your plans. This allows for early feedback and prevents wasted effort on impractical ideas.
2. The most valuable new datasets are those that test different capabilities than those already present. Bonus points if the task is similar to what users may face in the real world. Task ideas that would be great to add:
    - A task involving both language and  image modalities.
    - An image generation task (like stable diffusion)
    - A task involving audio (like whisper)
    - A task that requires knowledge preservation (checked, for instance, via an auxiliary test set)
    - Learning something completely new (e.g. a new language)
    - A reinforcement learning task (e.g. using [trl](https://github.com/huggingface/trl))
3. Training should be reasonably fast. Running dozens of experiments is impractical if each one takes multiple days and incurs high costs. Ideally, training should take a few hours at most on high-end consumer hardware.
4. The chosen base model should not be too large, to avoid VRAM constraints. Morevoer, if the base model is too powerful, there is little room for improvement through further fine-tuning.
5. Test scores should be informative and have a broad range:
   - Besides loss, there should ideally be at least one additional metric, such as accuracy.
   - Comparisons are not meaningful if all methods score near 0% or near 100%. The dataset should yield a range of scores to facilitate meaningful differentiation between methods.
6. The dataset should be publicly available and have a track record as a useful dataset. The license should permit the intended usage.

## Result dashboard

For convenience, we included a [Gradio](https://www.gradio.app/) app that shows the results of the experiments. It allows you to filter down the task and base model and show the experiment results for this selection. Give it a try!

This app requires additional packages to be installed, please install the packages listed in `requirements-app.txt`, e.g. via:

```sh
python -m pip install -r requirements-app.txt
```

To launch the demo, run:

```sh
python app.py
```
