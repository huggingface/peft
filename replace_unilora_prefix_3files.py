import os

target_files = [
    "src/peft/tuners/unilora/config.py",
    "src/peft/tuners/unilora/layer.py",
    "src/peft/tuners/unilora/model.py",
]

for fpath in target_files:
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = content.replace("UniLora_", "unilora_")
        if new_content != content:
            print(f"[MODIFIED] {fpath}")
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
    else:
        print(f"[WARNING] File not found: {fpath}")

print("Done.")

