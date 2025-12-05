# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import time

import aiohttp
import typer


def chat(
    msg: str,
    target: str = "lora1",
    max_tokens: int = 100,
    deterministic: bool = True,
    record: str = "",
    num_requests: int = 32,
):
    payload = {
        "model": target,
        "prompt": msg,
        "max_tokens": max_tokens,
    }
    if deterministic:
        payload = payload | {"temperature": 0, "top_p": 1, "top_k": 1}

    url = "http://localhost:8000/v1/completions"

    async def _request(session):
        async with session.post(url, json=payload) as response:
            response_json = await response.json()
            return response_json["choices"][0]["text"]

    async def run_concurrent():
        async with aiohttp.ClientSession() as session:
            tasks = [_request(session) for _ in range(num_requests)]
            return await asyncio.gather(*tasks)

    start_time = time.time()
    response_texts = asyncio.run(run_concurrent())
    end_time = time.time()

    print(f"Completed {num_requests} requests in {end_time - start_time:.2f} seconds")

    if record:
        with open(record, "w") as f:
            json.dump({"prompt": msg, "responses": response_texts}, f, indent=2)
    else:
        print(response_texts[0])


if __name__ == "__main__":
    typer.run(chat)
