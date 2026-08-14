import os
os.environ["HF_TOKEN"] = os.environ["HF_JOBS_TOKEN"]

from huggingface_hub import run_job, wait_for_job, fetch_job_logs

# Run the exact CI command: make tests_examples_multi_gpu
# But only run the prefix tuning test to save time
cmd = """set -e
source activate peft
cd /tmp
git clone --depth 1 https://github.com/huggingface/peft.git
cd peft
pip install -e . --no-deps 2>&1 | tail -2
echo "=== Running multi_gpu_tests filter on TestPrefixTuning ==="
CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 python -m pytest -m multi_gpu_tests tests/test_gpu_examples.py::TestPrefixTuning -xvs 2>&1
echo "=== Exit code: $? ==="
"""

print("Submitting CI reproduction job...")
job = run_job(
    image="huggingface/peft-gpu:latest",
    command=["bash", "-c", cmd],
    flavor="a10g-small",
    timeout=900,
)
print(f"Job ID: {job.id}")

final = wait_for_job(job_id=job.id, timeout=900)
print(f"Job status: {final.status.stage}")

for log in fetch_job_logs(job_id=job.id):
    print(log)