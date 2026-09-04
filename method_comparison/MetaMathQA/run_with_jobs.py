"""
Helper to run MetaMathQA experiments on a HF jobs runner.

The experiment path gives the path to the folder with the adapter_config.json
(and possibly train_params.json) relative to /method_comparison/MetaMathQA/experiments/
in the repository. Alternatively, if --upload is specified, it is assumed the experiment
exists locally (relative to the current working directory) and is uploaded to the
remote experiments folder.

Common use-cases:

* run_with_jobs.py --repo mygithubhandle:my_branch lora/llama-3.2-3B-rank32
  run an experiment from a custom repo/branch

* run_with_jobs.py --repo https://github.com/mygithubhandle/peft.git --branch my_branch lora/llama-3.2-3B-rank32
  same as above with explicit URL / branch

* run_with_jobs.py --repo huggingface:main --upload lora/custom-local-lora-exp
  run a locally modified experiment on the PEFT main branch

* run_with_jobs.py --code_bucket myuser/peft lora/llama-3.2-3B-rank32
  use the PEFT code from the myuser/peft bucket instead of cloning a git repo

"""

import os
import argparse
import subprocess

from base64 import b64encode

from huggingface_hub import run_job, Volume, cancel_job, fetch_job_logs

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=str, default="https://github.com/githubnemo/peft.git")
parser.add_argument("--branch", type=str)
parser.add_argument("--code_bucket", type=str, default=None, help="Bucket to use instead of git repo")
parser.add_argument("--upload", action="store_true", default=False)
parser.add_argument("experiment_path", type=str)
parser.add_argument("--flavor", type=str, default="a10g-large")
parser.add_argument("--debug", action="store_true", default=False)

args = parser.parse_args()

token = subprocess.run(
    ["hf", "auth", "token"], capture_output=True, text=True, check=True
).stdout.strip() or os.environ.get("HF_TOKEN", "")

if not token:
    print("No token, will not be able to load private or semi-private models.")

if "/" not in args.experiment_path:
    raise ValueError("experiment path must contain /, e.g. osf/llama-3.2-rank128")

volumes = []

if args.code_bucket:
    volumes.append(Volume(type="bucket", source=args.code_bucket, mount_path="/tmp/peft"))

if "@" not in args.repo and "://" not in args.repo:
    repo_parts = args.repo.split(":")
    args.repo = f"https://github.com/{repo_parts[0]}/peft.git"
    args.branch = repo_parts[1]

experiment_name = os.path.split(args.experiment_path)[-1]

if args.upload:
    adapter_config_path = os.path.join(args.experiment_path, "adapter_config.json")
    training_params_path = os.path.join(args.experiment_path, "training_params.json")

    adapter_config = ""
    training_params = ""

    if not os.path.exists(adapter_config_path):
        raise ValueError(f"No experiment config exists in {adapter_config_path}.")

    with open(adapter_config_path) as f:
        adapter_config = f.read()

    if os.path.exists(training_params_path):
        with open(training_params_path) as f:
            training_params = f.read()

cmd = (
    f"export HF_TOKEN={token} && "
    + "source activate peft && "
    + "pip uninstall mslk torchao -y -q 2>/dev/null; "  # TODO remove once this issue is resolved
    + (f"git clone {args.repo} /tmp/peft && " if not args.code_bucket else "")
    + "cd /tmp/peft && "
    + (f"git checkout {args.branch} && " if not args.code_bucket else "")
    + f"mkdir -p /tmp/peft/method_comparison/MetaMathQA/experiments/jobs/{experiment_name} && "
    + (
        f"echo '{b64encode(adapter_config)}' | base64 -d > /tmp/peft/method_comparison/MetaMathQA/experiments/{experiment_name}/adapter_config.json &&"
        if args.upload
        else ""
    )
    + (
        f"echo '{b64encode(training_params)}' | base64 -d > /tmp/peft/method_comparison/MetaMathQA/experiments/{experiment_name}/training_params.json &&"
        if args.upload and training_params
        else ""
    )
    + "pip install -e . --no-deps && "
    + "cd method_comparison/MetaMathQA && "
    + f"python run.py -v experiments/jobs/{experiment_name} --clean &&"
    + "cat temporary_results/*.json"
)

if args.debug:
    print(cmd)

job = run_job(
    image="huggingface/peft-gpu:latest",
    command=["bash", "-c", cmd],
    flavor=args.flavor,
    timeout=7200,
    volumes=volumes,
)
print(f"Job ID: {job.id}")
print(f"Status: {job.status}")

for log in fetch_job_logs(job_id=job.id, follow=True):
    print(log)

print(f"stopping job {job.id}...")
cancel_job(job_id=job.id)
