import re
import os
import json
import pickle as pkl
import numpy as np
import glob
import pandas as pd
from configs import DATA_DIR, PROMPTS_DIR, RESULT_DIR
from datasets import load_dataset
import nltk


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def get_mean_acc_models(input_dir, model_names, dataset_name="train"):
    _, labels = load_gsm8k_data(dataset_name=dataset_name)
    base_model_preds = []
    model_sample_count = []
    for model_n in model_names:
        results_dir = os.path.join(input_dir, model_n, dataset_name)
        all_files_id = [int(fn.split("_")[1]) for fn in
                        os.listdir(results_dir) if "npy" in fn]
        max_file_count = max(all_files_id)
        pred_path = os.path.join(input_dir, model_n, dataset_name,
                                 f"run_{max_file_count}_predictions.npy")
        pred_arr = np.load(pred_path)
        model_sample_count.append(max(all_files_id))
        avg_acc = []
        k = pred_arr.shape[1]
        for i in range(k):
            avg_acc.append(np.mean(pred_arr[:, i] == labels[:len(pred_arr)]) * 100)
        print(f"{model_n}: {np.mean(avg_acc):.2f} +- {2 * np.std(avg_acc)}")
        base_model_preds.append(pred_arr)
    min_size = min(model_sample_count)
    base_model_preds = np.concatenate([pred[:min_size] for pred in base_model_preds], axis=1)

    def majority_voting(in_arr):
        val, count = np.unique(in_arr, return_counts=True)
        return val[np.argmax(count)]

    ens_pred_flat = np.apply_along_axis(majority_voting, axis=1, arr=base_model_preds)
    acc = np.mean(labels[:min_size] == ens_pred_flat) * 100
    print(f"Majority Voting ALL: {acc:.2f}")


def load_gsm8k_prob_and_label(input_dir, model_names, num_samples=None,
                              num_runs=10, space_size=30, drop_non_exists=True, dataset_name="train"):
    # find number of samples for each model
    model_sample_count = []
    for model_n in model_names:
        results_dir = os.path.join(input_dir, model_n, dataset_name)
        all_files_id = [int(fn.split("_")[1]) for fn in
                        os.listdir(results_dir) if "npy" in fn]
        model_sample_count.append(max(all_files_id))
    min_size = min(model_sample_count)

    if num_samples is None:
        num_samples = min_size
    elif num_samples > min_size:
        model_n = [model_n for i, model_n in enumerate(model_names)
                   if model_sample_count[i] < min_size]
        raise ValueError(f"{' '.join(model_n)} models have less then {num_samples}")

    # load the maximum and then truncate
    all_pred = []
    for i, model_n in enumerate(model_names):
        pred_path = os.path.join(input_dir, model_n, dataset_name,
                                 f"run_{model_sample_count[i]}_predictions.npy")
        pred_arr = np.load(pred_path)[:num_samples, :num_runs]
        assert pred_arr.shape == (num_samples, num_runs)
        all_pred.append(pred_arr)
    all_pred_arr = np.concatenate(all_pred, axis=1)

    # extract probabilities for each model
    data_probs, solution_space = [], []
    for i in range(len(all_pred_arr)):
        sol_space = np.unique(all_pred_arr[i])
        probs_per_model = []
        for model_pred in all_pred:
            uni, counts = np.unique(model_pred[i], return_counts=True)
            count_dict = dict(zip(uni, counts))
            model_prob = []
            for j in sol_space:
                if j in count_dict.keys():
                    model_prob.append(count_dict[j] / sum(counts))
                else:
                    model_prob.append(0)
            probs_per_model.append(np.array(model_prob))
        # sort according to probabilities
        idx = np.argsort(np.sum(probs_per_model, axis=0))[::-1]
        probs_per_model = [prob[idx] for prob in probs_per_model]
        sol_space = sol_space[idx]
        data_probs.append(probs_per_model)
        solution_space.append(sol_space)

    # make each sample output to have the same space
    max_space_size = max([len(data[0]) for data in data_probs])
    if space_size < max_space_size:
        print("Truncating the space size")
    for i in range(len(data_probs)):
        if len(data_probs[i][0]) < space_size:
            pad_count = space_size - len(data_probs[i][0])
            pad_arr = np.zeros(pad_count)
            solution_space[i] = np.concatenate([solution_space[i], pad_arr])
            for j in range(len(data_probs[i])):
                data_probs[i][j] = np.concatenate([data_probs[i][j], pad_arr])
        else:
            for j in range(len(data_probs[i])):
                data_probs[i][j] = data_probs[i][j][:space_size]
            solution_space[i] = solution_space[i][:space_size]
    data_probs = np.stack(data_probs).reshape(num_samples, -1)
    solution_space = np.stack(solution_space)

    _, labels = load_gsm8k_data(dataset_name=dataset_name)
    labels = labels[:num_samples]

    # drop episodes that are not in the solution space (do this only for training)
    x, y = [], []
    for i in range(len(labels)):
        if labels[i] in solution_space[i]:
            y.append(np.argwhere(solution_space[i] == labels[i])[0].item())
        else:
            y.append(np.nan)
        x.append(data_probs[i])
    y, x = np.array(y), np.array(x)
    data = np.concatenate([x, y[:, None]], axis=1)

    if drop_non_exists:
        idx = ~np.isnan(y)
        data = data[idx]

    return data


def extract_ans_mmlu(in_arr, questions):
    pattern = r'[A-D]\.'
    pred_list = in_arr.tolist()
    options = ["A", "B", "C", "D"]
    extracted_data = []
    for i in range(len(pred_list)):
        inner_data = []
        for out in pred_list[i]:
            matches = re.findall(pattern, out)
            if len(matches) > 0:
                inner_data.append(matches[-1].split('.')[0])
            else:
                choices = questions[i].split("\n")[1:5]
                scores = [nltk.translate.bleu_score.sentence_bleu
                          ([out.split(" ")], choice.split(" "),
                           weights=(0.5, 0.5)) for choice in choices]
                idx = np.argmax(scores)
                inner_data.append(options[idx])
        extracted_data.append(inner_data)
    return np.array(extracted_data)


def load_mmlu_multi_run(model_names, dataset_name):
    data_dir = os.path.join(RESULT_DIR, "mmlu")
    # find number of samples for each model
    data = []
    choices = ["A", "B", "C", "D"]
    for model_n in model_names:
        arr_path = os.path.join(data_dir, model_n, dataset_name, f"latest_predictions.npy")
        outputs_path = os.path.join(data_dir, model_n, dataset_name, f"latest_outputs.pkl")
        predictions = np.load(arr_path)
        with open(outputs_path, "rb") as f:
            outputs = pkl.load(f)
        assert len(outputs) == len(predictions)

        labels = np.array(outputs)[:, 1]
        questions = np.array(outputs)[:, 0]

        extracted_data = extract_ans_mmlu(predictions, questions)
        probs = []
        for choice in choices:
            prob_choice = np.apply_along_axis(lambda x: np.sum(x == choice) / 10, 1, extracted_data)
            probs.append(prob_choice)
        data.append(np.stack(probs).T)
    data = np.concatenate(data, axis=1)
    labels = np.apply_along_axis(lambda x: choices.index(x), 0, labels[None, :])
    data = np.concatenate([data, labels[:, None]], axis=1)
    return data


def load_mmlu_prob_and_label(model_names):
    data_path = os.path.join(CUR_DIR, "other_results", "extracted")
    data_df_names = [os.path.basename(fname) for fname in
                     glob.glob(f"{data_path}/{model_names[0]}/*.csv")]
    data_df_names = sorted(data_df_names)

    choices = ["A", "B", "C", "D"]
    data = []
    for df_name in data_df_names:
        df_pred, df_label = [], []
        for m_name in model_names:
            dpath = f"{data_path}/{m_name}/{df_name}"
            data_df = pd.read_csv(dpath)
            a = data_df["prediction"].apply(lambda x: np.array(eval(x)))
            b = data_df["label"].apply(lambda x: choices.index(x))
            df_pred.append(np.exp(np.stack(a.values)))
            df_label.append(b.values)
        df_pred = np.concatenate(df_pred, axis=1)
        data.append(np.concatenate([df_pred, df_label[0][:, None]], axis=1))
    return data


def load_gsm8k_data(dataset_name):
    data_path = os.path.join(CUR_DIR, DATA_DIR, "gsm8k", f"{dataset_name}.jsonl")
    data_df = pd.read_json(data_path, lines=True)

    questions, labels = [], []
    pattern = r'####\s*(\S+)'
    for i, row in data_df.iterrows():
        matches = re.findall(pattern, row.answer)
        labels.append(float(matches[0].replace(",", "")))
        questions.append(row.question)

    return questions, labels


def load_searchqa_data(dataset_name):
    dataset = load_dataset("search_qa", "train_test_val",
                           split=dataset_name, trust_remote_code=True)
    return dataset["question"], dataset["answer"]

def load_xsum_data(dataset_name):
    dataset = load_dataset("EdinburghNLP/xsum", split=dataset_name, trust_remote_code=True)
    return dataset["document"], dataset["summary"]


