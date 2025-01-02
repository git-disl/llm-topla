import os
import argparse
import shutil
import time

import numpy as np
import pandas as pd
import pickle as pkl
import requests
import json

import os
import pandas as pd
from together import Together

import tqdm
from datasets import load_dataset
from openai import OpenAI
from ..configs import llm_domains, together_token, RESULT_DIR, PROMPTS_DIR


def search_ans(search_list, token_to_search):
    def new_line_search(start_idx, in_list):
        for j in range(200):
            if in_list[start_idx+j] == "\n":
                return start_idx + j

    found_idx = []
    for i in range(1, len(search_list)+1):
        for tkn in token_to_search:
            found = None
            if tkn in search_list[i-1]:
                found = i-1
            elif tkn[0] in search_list[i-1] and tkn[1] in search_list[i]:
                found = i-1

            if found is not None:
                nl_idx = new_line_search(found, search_list)
                found_idx.append((found, nl_idx))

    return found_idx


def get_mmlu_prompt_template(n_shot, dataset_type, sub_topic=None):
    with open(os.path.join(PROMPTS_DIR, "few_shot.json"), "r") as f:
        temp_dict = json.load(f)

    prompt = temp_dict["mmlu"][f"prompt_{dataset_type}"]
    if n_shot > 0:
        if sub_topic is not None:
            sub_file_name = sub_topic.replace(" ", "_") + "_dev"
            examples_data_path = f"data/mmlu/dev/{sub_file_name}.csv"
            ds_df = pd.read_csv(examples_data_path, header=None)
            ds_question_list = [f"{row[0]}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}" for idx, row in
                                ds_df.iterrows()]
            label_list = ds_df[ds_df.columns[-1]].tolist()
            example_txt = ""
            for question, label in zip(ds_question_list, label_list):
                example_txt += f"{question}\nCorrect answer:{label}\n"
            prompt = prompt.replace("{topic}", sub_topic)
            prompt = prompt.replace("{examples}", example_txt)

    return prompt


def load_mmlu_dataset(save_dir, dataset_type):
    data_path = f"{save_dir}/mmlu_{dataset_type}.csv"
    data_df = pd.read_csv(data_path)
    return data_df


def load_last_save(save_dir, load_last_save=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if load_last_save and os.listdir(save_dir):
        arr_path = os.path.join(save_dir, f"latest_predictions.npy")
        outputs_path = os.path.join(save_dir, f"latest_outputs.pkl")
        predictions = np.load(arr_path).tolist()
        with open(outputs_path, "rb") as f:
            outputs = pkl.load(f)
        assert len(outputs) == len(predictions)
        start_sample = len(outputs)
        print(f"Resuming at {len(outputs)}th step")
    else:
        predictions, outputs = [], []
        start_sample = 0

    return predictions, outputs, start_sample


def run(args):
    model_name = f"{llm_domains[args.llm_name]}/{args.llm_name}"
    ds_save_dir = "data/mmlu"
    dataset = load_mmlu_dataset(ds_save_dir, args.dataset_name)

    if args.num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = args.num_samples

    questions = dataset["question"][:args.num_samples].tolist()
    answers = dataset["label"][:args.num_samples].tolist()
    topics = dataset["topic"][:args.num_samples].tolist()

    client = Together(api_key=together_token)

    save_dir = os.path.join(RESULT_DIR, args.task_name, args.llm_name, args.dataset_name)
    predictions, outputs, start_sample = load_last_save(save_dir,
                                                        load_last_save=bool(args.load_last_save))
    questions = questions[start_sample:]
    answers = answers[start_sample:]

    count = start_sample
    backup_count = 0
    inference_time = 0
    for i in tqdm.tqdm(range(len(questions))):
        if len(questions[i]) >= 13000 and "Llama" in args.llm_name:
            print("Too long input truncating")
            questions[i] = questions[i][:13000]

        if args.dataset_name in ["test", "val"]:
            prompt_template = get_mmlu_prompt_template(n_shot=5,
                                                       dataset_type=args.dataset_name,
                                                       sub_topic=topics[i])
        else:
            prompt_template = get_mmlu_prompt_template(n_shot=0,
                                                       dataset_type="train")

        prompt = prompt_template.replace("{question}", questions[i])

        start_time = time.time()
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}],
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            # top_p=args.top_p,
            # top_k=args.top_k,
            logprobs=1,
            n=args.sample_per_pass,
        )
        elapsed_time = time.time() - start_time

        pred = []
        for choice in chat_completion.choices:
            answer_txt = choice.message.content
            pred.append(answer_txt)
            print(answer_txt)
        predictions.append(pred)
        outputs.append([questions[i], answers[i]])

        if count % args.save_freq == 0 or count == (num_samples - 1):
            backup_count += 1
            save_pred_arr = np.array(predictions)
            arr_path = os.path.join(save_dir, f"backup_{backup_count}_predictions.npy")
            outputs_path = os.path.join(save_dir, f"backup_{backup_count}_outputs.pkl")
            np.save(arr_path, save_pred_arr)
            with open(outputs_path, "wb") as f:
                pkl.dump(outputs, f)

            shutil.copy(arr_path, os.path.join(save_dir, "latest_predictions.npy"))
            shutil.copy(outputs_path, os.path.join(save_dir, "latest_outputs.pkl"))

            if backup_count > 1:
                backup_count = 0

        count += 1
        inference_time += elapsed_time

    avg_inf_time = inference_time / (i+1)
    print(f"Average inference time: {avg_inf_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM inference')
    parser.add_argument('--task_name', default="mmlu", type=str)
    parser.add_argument('--llm_name', default="phi-2",
                        choices=["gemma-2b-it", "gemma-7b-it", "Llama-2-13b-chat-hf", "Mixtral-8x7B-Instruct-v0.1",
                                 "Llama-2-70b-chat-hf", "Mistral-7B-Instruct-v0.1", "phi-2", "Llama-2-7b-chat-hf"])
    parser.add_argument("--dataset_name", default="test", choices=["train", "test"])
    parser.add_argument("--num_samples", default=None, type=int)
    parser.add_argument("--sample_per_pass", default=1, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--load_last_save", default=1, type=int, choices=[0, 1])
    parser.add_argument("--max_new_tokens", default=1024, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=40, type=int)
    arguments = parser.parse_args()
    print(arguments.llm_name)
    run(arguments)
