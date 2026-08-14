import os
os.environ["HF_TOKEN"] = os.environ["HF_JOBS_TOKEN"]

from huggingface_hub import run_job, wait_for_job, fetch_job_logs

# Run a few multi-GPU BNB tests before the prefix tuning test to see if state pollution causes the failure
# Also force 2 GPUs even though a10g-small has 1 GPU (won't help — need multi-GPU flavor)
# Instead, let's run the full multi_gpu_tests suite and check if the prefix tuning test fails
cmd = """set -e
source activate peft
cd /tmp
git clone --depth 1 https://github.com/huggingface/peft.git
cd peft
pip install -e . --no-deps 2>&1 | tail -2

echo "=== GPU count ==="
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo "=== Running TestPrefixTuning with multi_gpu marker ==="
CUDA_LAUNCH_BLOCKING=1 python -m pytest -m multi_gpu_tests tests/test_gpu_examples.py::TestPrefixTuning -xvs 2>&1

echo "=== Now running a BNB test before the prefix tuning test ==="
CUDA_LAUNCH_BLOCKING=1 python -m pytest -m multi_gpu_tests tests/test_gpu_examples.py::PeftBnbGPUExampleTests::test_causal_lm_training tests/test_gpu_examples.py::TestPrefixTuning::test_prefix_tuning_causal_lm_training_8bit_bnb -xvs 2>&1

echo "=== Exit code: $? ==="
"""

print("Submitting multi-test reproduction job...")
job = run_job(
    image="huggingface/peft-gpu:latest",
    command=["bash", "-c", cmd],
    flavor="a10g-small",
    timeout=1800,
)
print(f"Job ID: {job.id}")

final = wait_for_job(job_id=job.id, timeout=1800)
print(f"Job status: {final.status.stage}")

for log in fetch_job_logs(job_id=job.id):
    print(log)