import os
import pickle as pkl
import argparse
import numpy as np
import tqdm
import re
from openai import OpenAI

from ..helper import load_gsm8k_data, get_prompt_template
from ..configs import RESULT_DIR, deep_infra_token, llm_domains


def run(args):
    model_name = f"{llm_domains[args.llm_name]}/{args.llm_name}"
    questions, labels = load_gsm8k_data(dataset_name=args.dataset_name)
    prompt_template = get_prompt_template(task_name=args.task_name,
                                          llm_name=args.llm_name,
                                          n_shot=args.n_shot)
    prompts = []
    for question in questions:
        p = prompt_template.replace("{question}", question)
        prompts.append(p)

    if args.num_samples is None:
        num_samples = len(prompts)
    else:
        num_samples = args.num_samples
        prompts = prompts[:args.num_samples]

    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=deep_infra_token,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    save_dir = os.path.join(RESULT_DIR, args.task_name, args.llm_name, args.dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        load_last_save = 0
    else:
        load_last_save = args.load_last_save

    if load_last_save == 1 and os.listdir(save_dir):
        all_files_id = [int(fn.split("_")[1]) for fn in
                        os.listdir(save_dir) if "npy" in fn]
        id_max = max(all_files_id)
        arr_path = os.path.join(save_dir, f"run_{id_max}_predictions.npy")
        outputs_path = os.path.join(save_dir, f"run_{id_max}_outputs.pkl")
        predictions = np.load(arr_path).tolist()
        with open(outputs_path, "rb") as f:
            outputs = pkl.load(f)
        start_sample = id_max + 1
        prompts = prompts[start_sample:]
        print(f"Resuming at {id_max}th step")
    else:
        predictions, outputs = [], []
        start_sample = 0

    if args.sample_per_pass % 2 != 0:
        raise KeyError("sample per pass must be divisible by 2")

    answer_pattern_2 = r"[-+]?\d*\.?\d+"
    count = start_sample
    for in_prompt in tqdm.tqdm(prompts):
        inner_pred, inner_output = [], []
        for j in range(args.sample_per_pass // 2):
            chat_completion = openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Answer only the last question don't repeat the question" + in_prompt}],
                max_tokens=args.max_new_tokens,
                stop=["</s>", "####"],
                temperature=args.temperature,
                top_p=args.top_p,
                n=2
            )

            for choice in chat_completion.choices:
                answer_txt = choice.message.content
                # print(answer_txt)
                matches = re.findall(answer_pattern_2, answer_txt)
                if len(matches) > 0:
                    inner_pred.append(float(matches[-1]))
                else:
                    inner_pred.append(np.nan)
                inner_output.append(answer_txt)

        predictions.append(np.array(inner_pred))
        outputs.append(inner_output)

        if count % args.save_freq == 0 or count == (num_samples - 1):
            save_pred_arr = np.array(predictions)
            arr_path = os.path.join(save_dir, f"run_{count}_predictions.npy")
            outputs_path = os.path.join(save_dir, f"run_{count}_outputs.pkl")
            np.save(arr_path, save_pred_arr)
            with open(outputs_path, "wb") as f:
                pkl.dump(outputs, f)
        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM inference')
    parser.add_argument('--task_name', default="gsm8k", type=str)
    parser.add_argument('--llm_name', default="Llama-2-13b-chat-hf",
                        choices=["Mixtral-8x7B-Instruct-v0.1", "Llama-2-70b-chat-hf", "Llama-2-13b-chat-hf"])
    parser.add_argument("--dataset_name", default="train", choices=["train", "test"])
    parser.add_argument("--num_samples", default=None, type=int)
    parser.add_argument("--n_shot", default=5, type=int)
    parser.add_argument("--sample_per_pass", default=10, type=int)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--load_last_save", default=1, type=int, choices=[0, 1])
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=40, type=int)
    arguments = parser.parse_args()
    run(arguments)
