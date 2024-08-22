import json

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def writer(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

test_data_path = "testset.json"

test_data = read_json(test_data_path)

test_data_processed = []
for sample in test_data:
    test_data_processed.append({
        "instruction": sample["original_text"],
        "input": "",
        "output": "",
        "answer": str(sample["ans"]),
    })

writer(test_data_processed, "test.json")