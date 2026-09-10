# Generating data for training

    hf jobs uv run \
        --timeout 6h \
        --with torch \
        --with transformers \
        --with datasets \
        --flavor a100-large \
        -s HF_TOKEN="$(hf auth token)" \
        gen_selfdata.py \
            --device cuda --dataset HuggingFaceFW/finewiki --base_model Qwen/Qwen3.6-35B-A3B \
            --hub_id hubnemo/mtp-selfdata-qwen3.6-35b-a3b-finewiki --num_samples 10000

# Evaluating the MTP adapter

For evaluation we use `./eval_peft_mtp.py`, here are the corner stones of that evaluation:

* dataset is `hubnemo/tulu3-sft-mini`, a subsample of https://huggingface.co/datasets/allenai/tulu-3-sft-mixture that
provides a good mix of different domains so that we can test if the MTP adapter generalizes across domains and has a
speedup over the base model
* maximum generated tokens = 200 by default, long enough to reduce measurement noise and short enough to be speedy
* minimum generated tokens = num. MTP tokens so that every sample tests the MTP adapter's ability to generalize,
    otherwise it might be that the base model decides to generate EOS early, diminishing the sample's worth
* model dtype defaults to `bfloat16`
