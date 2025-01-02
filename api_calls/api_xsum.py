import time
import os
import argparse
import numpy as np
import pickle as pkl

import tqdm
from datasets import load_dataset
from openai import OpenAI
from ..configs import llm_domains, deep_infra_token, RESULT_DIR

# a = load_dataset("openai_humaneval")
# b = load_dataset("codeparrot/apps")


def load_last_save(save_dir, load_last_save=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if load_last_save and os.listdir(save_dir):
        all_files_id = [int(fn.split("_")[1]) for fn in
                        os.listdir(save_dir) if "npy" in fn]
        id_max = max(all_files_id)
        arr_path = os.path.join(save_dir, f"run_{id_max}_predictions.npy")
        outputs_path = os.path.join(save_dir, f"run_{id_max}_outputs.pkl")
        predictions = np.load(arr_path).tolist()
        with open(outputs_path, "rb") as f:
            outputs = pkl.load(f)
        start_sample = id_max + 1
        print(f"Resuming at {id_max}th step")
    else:
        predictions, outputs = [], []
        start_sample = 0

    return predictions, outputs, start_sample


def run(args):
    model_name = f"{llm_domains[args.llm_name]}/{args.llm_name}"
    # dataset = load_dataset("web_questions", split=args.dataset_name)
    dataset = load_dataset("EdinburghNLP/xsum",
                           split=args.dataset_name, trust_remote_code=True)

    if args.num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = args.num_samples

    questions = dataset["document"][:args.num_samples]
    answers = dataset["summary"][:args.num_samples]

    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=deep_infra_token,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    save_dir = os.path.join(RESULT_DIR, args.task_name, args.llm_name, args.dataset_name)
    predictions, outputs, start_sample = load_last_save(save_dir,
                                                        load_last_save=bool(args.load_last_save))
    questions = questions[start_sample:]
    answers = answers[start_sample:]

    count = start_sample
    inference_time = 0
    for i in tqdm.tqdm(range(len(questions))):
        if len(questions[i]) >= 13000 and "Llama" in args.llm_name:
            print("Too long input truncating")
            questions[i] = questions[i][:13000]

        start_time = time.time()
        chat_completion = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Try your best to summarize the main content of the given document."
                                              " And generate a short summary in 1 sentence only.\n Summary:"},
                {"role": "user", "content": questions[i]}],
            max_tokens=args.max_new_tokens,
            stop=["</s>"],
            temperature=args.temperature,
            top_p=args.top_p,
            n=1
        )
        elapsed_time = time.time() - start_time
        inference_time += elapsed_time

        for choice in chat_completion.choices:
            answer_txt = choice.message.content
            predictions.append(answer_txt)
        outputs.append([questions[i], answers[i]])

        if count % args.save_freq == 0 or count == (num_samples - 1):
            save_pred_arr = np.array(predictions)
            arr_path = os.path.join(save_dir, f"run_{count}_predictions.npy")
            outputs_path = os.path.join(save_dir, f"run_{count}_outputs.pkl")
            np.save(arr_path, save_pred_arr)
            with open(outputs_path, "wb") as f:
                pkl.dump(outputs, f)
        count += 1

    avg_inf_time = inference_time / args.num_samples
    print(f"Average inference time: {avg_inf_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM inference')
    parser.add_argument('--task_name', default="xsum", type=str, choices=["search_qa"])
    parser.add_argument('--llm_name', default="Mistral-7B-Instruct-v0.1",
                        choices=["gemma-7b-it", "Llama-2-13b-chat-hf", "Mixtral-8x7B-Instruct-v0.1", "Llama-2-70b-chat-hf", "Mistral-7B-Instruct-v0.1"])
    parser.add_argument("--dataset_name", default="train", choices=["train", "test"])
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--sample_per_pass", default=10, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--load_last_save", default=1, type=int, choices=[0, 1])
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=40, type=int)
    arguments = parser.parse_args()
    print(arguments.llm_name)
    run(arguments)





