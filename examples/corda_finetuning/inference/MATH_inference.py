import argparse
import sys

import jsonlines
import util
from vllm import LLM, SamplingParams


MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def process_results(doc, completion, answer):
    split_ans = completion.split("The answer is: ")
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split(".\n")[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if util.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {"question": doc, "output": completion, "answer": answer}
        invalid_outputs.append(temp)
        return False


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print("promt =====", problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item["output"]
            temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print("total length ===", len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print("lenght ====", len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
    print("sampleing =====", sampling_params)
    llm = LLM(
        model=model, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.8, trust_remote_code=True
    )
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_hendrycks_math_ins, hendrycks_math_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(
        zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)
    ):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    acc = sum(results) / len(results)
    print("len invalid outputs ====", len(invalid_outputs), ", valid_outputs===", invalid_outputs)
    print("start===", start, ", end====", end)
    print("length====", len(results), ", acc====", acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=0)  # model path
    parser.add_argument("--data_file", type=str, default="data/MATH_test.jsonl")  # data path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=50)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(
        model=args.model,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )
