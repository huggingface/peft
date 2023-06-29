import json, os
from pathlib import Path
from datetime import date
from tabulate import tabulate

failed = []
passed = []

group_info = []

total_num_failed = 0
empty_file = False or len(list(Path().glob("*.log"))) == 0
for log in Path().glob("*.log"):
    section_num_failed = 0
    with open(log, "r") as f:
        nb_lines = sum(1 for _ in f)
        for i, line in f:
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
        if nb_lines == 0:
            empty_file = True
    group_info.append([str(log), section_num_failed, failed])
    os.remove(log)
    failed = []
no_error_payload = {
    "type": "section",
    "text": {
        "type": "plain_text",
        "text": "🌞 There were no failures!" if not empty_file else "Something went wrong - please check GH action results.",
        "emoji": True
    }
}

message = ""
payload = [
    {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": "🤗 Results of the {} PEFT scheduled tests.".format(os.environ.get("TEST_TYPE", "")),
        }
    },
]
if total_num_failed > 0:
    for name, num_failed, failed_tests in group_info:
        if num_failed > 0:
            if num_failed == 1:
                message += f"*{name}: {num_failed} failed test*\n"
            else:
                message += f"*{name}: {num_failed} failed tests*\n"
            failed_table = []
            for test in failed_tests:
                failed_table.append(test[0].split("::"))
            failed_table = tabulate(failed_table, headers=["Test Location", "Test Case", "Test Name"], showindex="always", tablefmt="grid", maxcolwidths=[12, 12, 12])
            message += '\n```\n' +failed_table + '\n```'
    print(f'### {message}')
else:
    payload.append(no_error_payload)
    

if os.environ.get("TEST_TYPE", "") != "":
    from slack_sdk import WebClient

    if len(message) != 0:
        md_report = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message
            },
        }
        payload.append(md_report)
        action_button = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*For more details:*"
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/peft/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }
        payload.append(action_button)

    date_report = {
        "type": "context",
        "elements": [
            {
                "type": "plain_text",
                "text": f"Nightly {os.environ.get('TEST_TYPE')} test results for {date.today()}",
            },  
        ],
    }
    payload.append(date_report)

    print(payload)

    client = WebClient(token=os.environ.get("SLACK_API_TOKEN"))
    client.chat_postMessage(channel="#peft-ci-daily", text=message, blocks=payload)
