import argparse
import re
import sys

import jsonlines
from fraction import Fraction
from grader import math_equal
from vllm import LLM, SamplingParams


MAX_INT = sys.maxsize


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split("The answer is: ")
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
        if match:
            if "/" in match.group():
                denominator = match.group().split("/")[1]
                numerator = match.group().split("/")[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == "0":
                        return round(float(numerator.replace(",", "")))
                    else:
                        frac = Fraction(match.group().replace(",", ""))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(",", "")) == float("inf"):
                    return None
                return round(float(match.group().replace(",", "")))
        else:
            return None
    else:
        return None


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


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print("promt =====", problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item["answer"].split("#### ")[1]
            temp_ans = int(temp_ans.replace(",", ""))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print("lenght ====", len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1024, stop=stop_tokens)
    print("sampleing =====", sampling_params)
    llm = LLM(
        model=model, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.8, trust_remote_code=True
    )  #
    # llm = transformers.AutoModelForCausalLM.from_pretrained(
    #    model,
    #    #torch_dtype=torch.bfloat16,
    #    device_map="auto",
    #    trust_remote_code=True
    # )
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer) or math_equal(y_pred, prompt_answer))
        else:
            result.append(False)
            temp = {"question": prompt, "output": completion, "answer": prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print("len invalid outputs ====", len(invalid_outputs), ", valid_outputs===", invalid_outputs)
    print("start===", start, ", end====", end)
    print("gsm8k length====", len(result), ", gsm8k acc====", acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default="data/gsm8k_test.jsonl")  # data path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=60)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(
        model=args.model,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )
