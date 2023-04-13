import json, os
from pathlib import Path
from datetime import date

failed = []
passed = []

group_info = []

total_num_failed = 0
for log in Path().glob("*.log"):
    section_num_failed = 0
    with open(log, "r") as f:
        for line in f:
            line = json.loads(line)
            if line.get("nodeid", "") != "":
                test = line["nodeid"]
                if line.get("duration", None) is not None:
                    duration = f'{line["duration"]:.4f}'
                    if line.get("outcome", "") == "failed":
                        section_num_failed += 1
                        failed.append([test, duration, log.name.split('_')[0]])
                        total_num_failed += 1
                    else:
                        passed.append([test, duration, log.name.split('_')[0]])
    group_info.append([str(log), section_num_failed, failed])
    failed = []
message = ""
if total_num_failed > 0:
    for name, num_failed, failed_tests in group_info:
        if num_failed > 0:
            if num_failed == 1:
                message += f"*{name}: {num_failed} failed test*\n"
            else:
                message += f"*{name}: {num_failed} failed tests*\n"
            failed_table = '| Test Location | Test Class | Test Name | PyTorch Version |\n|---|---|---|---|\n| '
            for test in failed_tests:
                failed_table += ' | '.join(test[0].split("::"))
            failed_table += f" | {test[2]} |"
            message += failed_table
    print(f'### {message}')
else:
    message = "No failed tests! 🤗"
    print(f'## {message}')

if os.environ.get("TEST_TYPE", "") != "":
    from slack_sdk import WebClient
    message = f'*Nightly {os.environ.get("TEST_TYPE")} test results for {date.today()}:*\n{message}'

    client = WebClient(token=os.environ['SLACK_API_TOKEN'])
    client.chat_postMessage(channel='#accelerate-ci-daily', text=message)