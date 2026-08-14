import os
os.environ["HF_TOKEN"] = os.environ["HF_JOBS_TOKEN"]

from huggingface_hub import list_jobs, fetch_job_logs, inspect_job

# List recent jobs
all_jobs = []
for j in list_jobs():
    all_jobs.append(j)
    if len(all_jobs) >= 10:
        break

for j in all_jobs:
    print(f"{j.id} | {j.status.stage}")
    try:
        info = inspect_job(job_id=j.id)
        if info.status.stage in ("COMPLETED", "ERROR"):
            logs = fetch_job_logs(job_id=j.id)
            log_text = "\n".join(logs)
            # Just show last 50 lines
            lines = log_text.strip().split("\n")
            for line in lines[-50:]:
                print(f"  {line}")
    except Exception as e:
        print(f"  Error: {e}")
    print()