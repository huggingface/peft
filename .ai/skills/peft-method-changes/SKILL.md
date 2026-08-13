---
name: peft-method-changes
description: "Use when modifying an existing PEFT method (its forward method, parameters/buffers, state_dict handling, or config defaults) to ensure that existing checkpoints keep working."
version: 1.0.0
license: Apache-2.0
tags: [peft, huggingface, refactor, contributions, backward-compatibility]
---

# Making changes to existing PEFT methods

When making changes to a PEFT method that:

- affects the `forward` method
- adds/removes parameters or buffers
- changes the logic of creating or loading the `state_dict`
- adds config options or changes the default behavior of existing ones

we run the risk of invalidating existing checkpoints. This skill describes the measures to take to prevent such backward-incompatible changes.

## Examples of when the precautions are needed

- Rewriting the `forward` method to require less memory or run faster.
- Changing the algorithm, e.g. because a follow up paper was released.
- Swapping the quantization backend, e.g. from one package to another.
- Improving the state dict to make the checkpoint smaller.

## When this is not needed

- When adding a completely new PEFT method or when adding a new, non-default option to an existing PEFT method.
- When the change is obviously mathematically the same but can have numeric deviations, e.g. because of floating point math. In that case, a mention in the release notes is sufficient.
- Forward-compatibility: It is not a requirement that older PEFT versions are compatible with checkpoints created with newer PEFT versions.

## Why the precautions are needed

We have to ensure that if a user trained a certain checkpoint, it keeps on working even after they update PEFT (backward-compatibility). Otherwise, many hours and resources for training could be wasted. Even worse, subtle changes could be hard to detect, resulting in the user not noticing that their PEFT model suddenly misbehaves in production. We want to ensure that users never feel the need to pin the upper PEFT version.

Moreover, we want an old training script to create reproducible outputs when re-run with the new PEFT version. Given the general limits to reproducibility, we can't expect the final checkpoint to be 100% identical, but it should be reasonably close.

## Necessary precautions

This section describes which steps are necessary to ensure that the change is safe.

### Unit testing

As with any change in PEFT, ensure that all unit tests keep passing: `pytest tests/ -k <peft-method-name>`.

### Regression testing

Run `pytest tests/regression/test_state_dict.py --regression -k <peft-method-name>` to ensure that old checkpoints keep on working. If the PEFT method (or the specific parametrization of said method) is not covered by this regression test, please ping the PEFT maintainers and ask them to add a test case and regression artifact.

### State dict updates

In general, it is possible to make changes to the `state_dict` structure, e.g. moving a PEFT module into a sub-module. This invalidates old `state_dict`s but if this can be reliably detected and fixed, this is fine. Here is an example of such a fix for a change in DoRA:

- https://github.com/huggingface/peft/blob/a429b594910844a21634114c5776c3ef5d0217a4/src/peft/utils/save_and_load.py#L194-L205
- https://github.com/huggingface/peft/blob/a429b594910844a21634114c5776c3ef5d0217a4/src/peft/utils/save_and_load.py#L922-L932

### Config backward compatibility

If the change adds a new field to the method's config class, remember that checkpoints created before the change have an `adapter_config.json` that lacks this field. When such a checkpoint is loaded, the field is filled in with its default value, so the default must reproduce the behavior from before the change.

### Run a PEFT benchmark and ensure that the results are reasonably similar

This is to ensure that over a longer training run, changes that affect the output don't accumulate and lead to different results. The recommended benchmark to run is the [MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) (note that this requires a CUDA GPU). Run the benchmark twice on the same machine, once on the `main` branch and once on your branch, and compare the two runs. The published results (found [in the results directory](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA/results) and browsable in the [method comparison app](https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison)) can serve as an additional sanity check, but since they were created on different hardware and with different library versions, larger deviations are expected there. Notably, it is expected that the `train_loss` stays within close proximity of the run on `main`, say, within 1e-4 tolerance.

Some metrics like test accuracy have a higher variance, say +/- 1 percentage point. Variations here are expected, but if they are larger than that, take note.

Other metrics like memory consumption or runtime can change too, especially if the suggested change has the intent of improving those exact metrics. But if the change should not affect those metrics, double check that they are also reasonably close.

### Dedicated testing script

Even with all the precautions described above, backward incompatible changes may slip through because they're not being tested. The number of possible edge cases is just too large to cover all of them this way. Therefore, one additional step is to create a dedicated testing script that ensures that the change does not affect the output. This script should take into account the following:

- always use a 'real' model, not a toy model (it can be a "tiny random" model though, like the ones used in testing)
- always test that the outputs remain the same
- always check that with and without merging, the outputs are identical
- always check with different dtypes (float32, float16, bfloat16), remember to set `autocast_adapter_dtype=False` so that the given dtype is used
- if the change concerns one specific layer type, e.g. `nn.Conv2d`, ensure that the tested model uses that layer type and that the layer type is targeted
- if quantization is involved, use this quantization method
- if a specific option is changed, e.g. `LoraConfig(use_foobar=True)`, ensure to use that option
- if the change only affects training mode (e.g. when it involves dropout), test this specifically

The testing script should work as follows: On the first run, using the existing `main` branch, it loads the model, applies the PEFT method with different parametrizations, generates an output and a checkpoint, and saves those to a temporary directory. For the second run, first switch to your branch, then run the script again. The script should find the existing checkpoint and expected output, load the checkpoint, create a new output, and ensure that the new output and the old output are reasonably close.

The following code block gives you a rough template for the script used for checking. Take it as inspiration, not gospel.

```python
# note: this directory needs to survive between the run on main and the run on your branch
PATH = Path(tempfile.gettempdir()) / "<peft-method-to-test>"
MODEL_ID = ...  # typically a transformers or diffusers model
SEED = 0
ATOL = 1e-4
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DEVICES = ["cpu", "cuda:0"]
TEST_CASES = {
    "default parameters": MyPeftConfig(),
    "with foobar turned on": MyPeftConfig(foobar=True),
    # add more cases as needed
}

def compute_output(model, device, dtype):
    ...

def compare_outputs(output_before, output_after):
    ...

def check_case(test_case_name, config, dtype, device):
    torch.manual_seed(SEED)
    output_path = PATH / f"..."      # the expected outputs, with and without merging
    checkpoint_path = PATH / f"..."  # the saved adapter checkpoint
    if not checkpoint_path.exists():
        print(f"No outputs found at {checkpoint_path}, creating them now")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
        model = get_peft_model(model, config)
        output = compute_output(model, device, dtype)
        model.merge_adapter()
        output_merged = compute_output(model, device, dtype)
        model.unmerge_adapter()
        # store both outputs and save the checkpoint
        ...
        return

    print(f"Outputs found at {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    output_before, output_before_merged = ...  # load the stored outputs
    output_after = compute_output(model, device, dtype)
    model.merge_adapter()
    output_after_merged = compute_output(model, device, dtype)
    return (
        compare_outputs(output_before, output_after),
        compare_outputs(output_before_merged, output_after_merged),
    )

def report_results(results):
    ...

def main():
    results = []
    for test_case_name, config in TEST_CASES.items():
        for dtype in DTYPES:
            for device in DEVICES:
                result = check_case(test_case_name, config, dtype, device)
                results.append((test_case_name, dtype, device, result))
    report_results(results)
```

## Reporting the results

In the PR description of this change, mention all the tests you've done and what the outcome was. For the benchmark run, show the relevant metrics side by side, e.g. in a markdown table. The dedicated testing script is *not* added to the repository, but attach it to the PR so that others can run it and for documentation purposes.

## Summary checklist

- [ ] Unit tests pass: `pytest tests/ -k <peft-method-name>`
- [ ] Regression tests pass: `pytest tests/regression/test_state_dict.py --regression -k <peft-method-name>`
- [ ] If the `state_dict` structure changed: old checkpoints are reliably detected and fixed on loading
- [ ] If new config fields were added: their defaults reproduce the previous behavior
- [ ] MetaMathQA benchmark results on `main` and on the branch are reasonably similar
- [ ] Dedicated testing script confirms identical outputs (merged and unmerged, across dtypes)
- [ ] PR description reports all results; the testing script is attached to the PR
