# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser


def create_test_prompts(args) -> list[tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    with open(f"{os.getcwd()}/prompt.json") as f:
        prompt_pool = json.load(f)
    prompt = prompt_pool[str(args.input_len)]
    sampling_params= SamplingParams(
        n=1,
        temperature=0,
        top_k=-1,
        max_tokens=args.output_len
    )
    return [(prompt, sampling_params)]


def process_requests(engine: LLMEngine,
                     test_prompts: list[tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts(args)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument("--input-len", type=int, default=32,
        help="input prompt length")
    parser.add_argument("--output-len", type=int, default=32,
        help="output generate length")
    args = parser.parse_args()
    print(f"==>vllm::args,{args}")
    main(args)
