# SPDX-License-Identifier: Apache-2.0
import asyncio
from openai import AsyncOpenAI

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MODEL="ibm-research/PowerMoE-3b"

async def create_generate_tasks(loop, api_clients, prompts):
    async def generate(client, prompt):
        # print(f"start run: {prompt}")
        response = await client.completions.create(
            model=MODEL,
            prompt=prompt,
            stream=False,
        )
        reply = response.choices[0].text
        print(f"{bcolors.WARNING}prompt: {bcolors.ENDC}{prompt}\n{bcolors.WARNING}reply: {bcolors.ENDC}{reply}\n")
        # print(f"prompt: {prompt}\nreply: {reply}\n{response}")
        return response

    tasks = []
    for i, prompt in enumerate(prompts):
        # slow request
        # await asyncio.sleep(1)
        cidx = i % len(api_clients)
        # print(f"try to create task with prompt: {prompt} for client : {cidx}")
        client = api_clients[cidx]
        tasks.append(
            loop.create_task(
                generate(client, prompt)))
    await asyncio.wait(tasks)

if __name__ == "__main__":
    SERVE_PORTS = ["8801", "8802"]
    api_clients = []
    for dp_rank, port in enumerate(SERVE_PORTS):
        client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1"
        )
        api_clients.append(client)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "This is a test",
    ]
    # assert len(prompts) % len(api_clients) == 0

    loop = asyncio.get_event_loop()
    loop.run_until_complete(create_generate_tasks(loop, api_clients, prompts))
    loop.close()