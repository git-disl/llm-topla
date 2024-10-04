import os
import argparse
import re
import pickle as pkl
import tqdm


from configs import RESULT_DIR, model_names_dict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import nltk

from helper import load_gsm8k_data, load_searchqa_data
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.metrics.scores import f_measure


class MyDataset(Dataset):
    def __init__(self, tokenized_inputs, labels, global_attention_tokens=None, negative_inputs=None):
        self.tokenized_inputs = tokenized_inputs
        self.labels = labels
        self.global_attention_tokens = global_attention_tokens
        self.negative_inputs = negative_inputs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokenized_inputs[idx].ids)
        attention_mask = torch.tensor(self.tokenized_inputs[idx].attention_mask)
        global_attentions = []
        start = False
        for i in input_ids:
            if start:
                if i == 50266:
                    start = False
                global_attentions.append(1)
            else:
                if i == 50265:
                    start = True
                global_attentions.append(0)
        global_attentions = torch.tensor(global_attentions)
        label = self.labels[idx]
        return_dict = {'input_ids': input_ids,
                       "labels": label,
                       'attention_mask': attention_mask,
                       'global_attention_mask': global_attentions}
        return return_dict


def load_model_outputs(model_names, input_dir, dataset_name, sample_percentage=None, k=1):
    task_name = os.path.basename(input_dir)
    model_sample_count = []
    for model_n in model_names:
        results_dir = os.path.join(input_dir, model_n, dataset_name)
        all_files_id = [int(fn.split("_")[1]) for fn in
                        os.listdir(results_dir) if "npy" in fn]
        model_sample_count.append(max(all_files_id))
    min_size = min(model_sample_count)
    num_samples = min_size

    if sample_percentage is not None:
        assert 1 >= sample_percentage > 0
        num_samples = int(num_samples * sample_percentage)

    model_outputs = []
    for model_n in model_names:
        results_dir = os.path.join(input_dir, model_n, dataset_name)
        all_files_id = [int(fn.split("_")[1]) for fn in
                        os.listdir(results_dir) if "npy" in fn]
        max_file_count = max(all_files_id)

        if task_name == "gsm8k":
            output_path = os.path.join(results_dir, f"run_{max_file_count}_outputs.pkl")
            with open(output_path, "rb") as f:
                outputs = pkl.load(f)
        else:
            pred_path = os.path.join(input_dir, model_n, dataset_name,
                                     f"run_{max_file_count}_predictions.npy")
            outputs = np.load(pred_path)

        outputs = outputs[:num_samples]
        model_outputs.append(outputs)

    if task_name == "gsm8k":
        model_outputs = np.array(model_outputs)[:, :, :k]
        questions, labels = load_gsm8k_data(dataset_name=dataset_name)
    else:
        questions, labels = load_searchqa_data(dataset_name=dataset_name)
        model_outputs = np.array(model_outputs)
    questions, labels = questions[:num_samples], labels[:num_samples]

    return model_outputs, questions, labels


def load_mmlu_outputs(model_names):
    from helper import extract_ans_mmlu
    data = []
    choices = ["A", "B", "C", "D"]
    for model_n in model_names:
        arr_path = os.path.join(RESULT_DIR, "mmlu", model_n, "test", f"latest_predictions.npy")
        outputs_path = os.path.join(RESULT_DIR, "mmlu", model_n, "test", f"latest_outputs.pkl")
        predictions = np.load(arr_path)
        with open(outputs_path, "rb") as f:
            outputs = pkl.load(f)
        assert len(outputs) == len(predictions)

        labels = np.array(outputs)[:, 1]
        questions = np.array(outputs)[:, 0]
        data.append(predictions[:, :1])
    data = np.stack(data)
    return data, questions, labels


def calc_metric(labels, pred_arr):
    blue_scores, em_scores, F1 = [], [], []
    for i in range(len(labels)):
        pred = pred_arr[i].strip().split(" ")[:4]
        blue_scores.append(nltk.translate.bleu_score.
                           sentence_bleu([labels[i].split(" ")], pred,
                                         weights=(1, 0, 0, 0)))

        reference_set = set(labels[i].split(" "))
        test_set = set(pred)
        F1.append(f_measure(reference_set, test_set))

        pred = " ".join(pred)
        ans = re.sub(r"[()]", "", labels[i]).split(" ")

        txt_wo_prn = re.sub(r'\([^)]*\)', '', labels[i])
        txt_with_prn = ans
        if txt_wo_prn == pred or txt_with_prn == pred:
            em_scores.append(1)
        else:
            em_scores.append(0)

    return np.mean(blue_scores), np.mean(em_scores), np.mean(F1)


def get_mean_acc_models(input_dir, model_names, dataset_name="train"):
    _, labels = load_searchqa_data(dataset_name=dataset_name)
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

        scores = calc_metric(labels[:max(all_files_id)], pred_arr)
        print(f"{model_n}: BLUE-1: {scores[0]:.4f} EM: {scores[1]:.4f} F1: {scores[2]:.4f}")
        base_model_preds.append(pred_arr)
    min_size = min(model_sample_count)
    base_model_preds = np.stack([pred[:min_size] for pred in base_model_preds], axis=1)

    def majority_voting(in_arr):
        val, count = np.unique(in_arr, return_counts=True)
        return val[np.argmax(count)]

    ens_pred_flat = np.apply_along_axis(majority_voting, axis=1, arr=base_model_preds)
    scores = calc_metric(labels[:min_size], ens_pred_flat)
    print(f"Majority Voting ALL: BLUE-1: {scores[0]:.4f} EM: {scores[1]:.4f} F1: {scores[2]:.4f}")


def tokenize_inputs(tokenizer, in_data, questions, in_label, task_name, skip_model_outs=False):
    if len(in_data.shape) == 3:
        M, N, K = in_data.shape
    elif len(in_data.shape) == 2:
        M, N = in_data.shape
        in_data = np.expand_dims(in_data, -1)
        K = 1
    else:
        M, N, K = (1, 0, 0)

    data = []
    for i in range(N):
        # create an input
        temp = [f"[BOQ]{questions[i]}[EOQ]"]
        if not skip_model_outs:
            for j in range(M):
                for k in range(K):
                    in_sentences = in_data[j, i, k].strip().replace("####", "").split(".")[-6:]
                    candidate_txt = "".join(in_sentences)
                    temp.append(f"[BOC{j}]{candidate_txt}[EOC{j}]")
        data.append("".join(temp))

    # add new tokens
    new_tokens = []
    num_added = 0
    vocab = tokenizer.get_vocab()
    for i in range(M):
        if f"[BOQ]" not in vocab and f"[BOQ]" not in new_tokens:
            new_tokens.append("[BOQ]")
            num_added += 1
        if f"[EOQ]" not in vocab and f"[EOQ]" not in new_tokens:
            new_tokens.append("[EOQ]")
            num_added += 1
        if f"[BOC{i}]" not in vocab and f"[BOC{i}]" not in new_tokens:
            new_tokens.append(f"[BOC{i}]")
            num_added += 1
        if f"[EOC{i}]" not in vocab and f"[EOC{i}]" not in new_tokens:
            new_tokens.append(f"[EOC{i}]")
            num_added += 1
    # num_added_toks = tokenizer.add_tokens(new_tokens)
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print("We have added", num_added_toks, "tokens")
    new_token_ids = [tokenizer.encode(tkn)[1] for tkn in new_tokens]

    model_inputs = tokenizer(data, padding="longest", max_length=16000, truncation=True, return_tensors="pt")
    # model_inputs = tokenizer(data, padding="max_length", max_length=4096, truncation=True, return_tensors="pt",
    #                          add_special_tokens=True)

    if task_name == "gsm8k":
        labels = []
        for lbl in in_label:
            lbl_txt = f"The answer is {int(lbl)}."
            labels.append(lbl_txt)
        lbl_ids = tokenizer(labels, padding="longest", max_length=4096, truncation=True, return_tensors="pt")
    elif task_name == "mmlu":
        labels = []
        for lbl in in_label:
            lbl_txt = f"The answer is {lbl}."
            labels.append(lbl_txt)
        lbl_ids = tokenizer(labels, padding="longest", max_length=4096, truncation=True, return_tensors="pt")
    else:
        lbl_ids = tokenizer(in_label, padding="longest", max_length=512, truncation=True, return_tensors="pt")

    return model_inputs, lbl_ids, new_token_ids


def extract_answer(tokenizer, prediction, task_name):
    batch_size = prediction.shape[0]
    pred = []
    for i in range(batch_size):
        answer_txt = tokenizer.decode(prediction[i], skip_special_tokens=True)
        if task_name == "mmlu":
            matches = re.findall(r'[A-D]', answer_txt)
            if matches:
                pred.append(matches[0])
            else:
                pred.append("")
        elif task_name == "gsm8k":
            matches = re.findall(r"[-+]?\d*\.?\d+", answer_txt)
            if matches:
                pred.append(float(matches[0]))
            else:
                pred.append(.0)
        else:
            pred.append(answer_txt.strip())
    return pred


def test_loop(model, tokenizer, eval_dataloader, device, task_name, mode="Validation", return_outputs=False):
    model.eval()
    predictions, labels = [], []
    progress_bar = tqdm.tqdm(range(len(eval_dataloader)))
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        predictions.append(extract_answer(tokenizer, pred, task_name))
        labels.append(extract_answer(tokenizer, batch["labels"], task_name))
        progress_bar.update(1)
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    if task_name in ["gsm8k", "mmlu"]:
        acc = np.mean(labels == predictions)
        print(f"{mode} Accuracy: {acc:.4f}")
        scores = [acc]
    elif task_name == "search_qa":
        scores = calc_metric(labels, predictions)
        print(f"{mode} BLUE-1: {scores[0]:.4f} EM: {scores[1]:.4f} F1: {scores[2]:.4f}")
    else:
        raise KeyError

    if return_outputs:
        return scores, predictions, labels
    else:
        return scores

def main(args):
    model_names = [model_names_dict[args.task_name][int(idx)] for idx in args.model_ids]
    print(args.model_ids, model_names)
    input_dir = os.path.join(RESULT_DIR, args.task_name)
    ens_model_n = "allenai/led-base-16384"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.task_name == "mmlu":
        outs, questions, lbl = load_mmlu_outputs(model_names)
        print(outs.shape, questions.shape, lbl.shape)
        train_size = int((len(questions) * 0.7) * args.train_percentage)
        test_size = int(len(questions) * 0.3)
        train_outs, train_q, train_lbl = outs[:, :train_size], questions[:train_size], lbl[:train_size]
        test_outs, test_q, test_lbl = outs[:, -test_size:], questions[-test_size:], lbl[-test_size:]
    else:
        train_outs, train_q, train_lbl = \
            load_model_outputs(model_names, input_dir, dataset_name="train", k=args.num_k,
                               sample_percentage=float(args.train_percentage))
        test_outs, test_q, test_lbl = \
            load_model_outputs(model_names, input_dir, dataset_name="test", k=args.num_k)

    tokenizer = AutoTokenizer.from_pretrained(ens_model_n)
    train_inputs, train_labels, new_token_ids = tokenize_inputs(tokenizer, train_outs, train_q, train_lbl,
                                                                skip_model_outs=bool(args.skip_model_outputs),
                                                                task_name=args.task_name)
    test_inputs, test_labels, _ = tokenize_inputs(tokenizer, test_outs, test_q, test_lbl,
                                                  skip_model_outs=bool(args.skip_model_outputs),
                                                  task_name=args.task_name)

    num_train_samples = len(train_inputs.data["input_ids"])
    train_size = int(num_train_samples * 0.7)

    train_dataset = MyDataset(train_inputs[:train_size], train_labels.input_ids[:train_size],
                              global_attention_tokens=new_token_ids)
    val_dataset = MyDataset(train_inputs[train_size:], train_labels.input_ids[train_size:],
                            global_attention_tokens=new_token_ids)
    test_dataset = MyDataset(test_inputs, test_labels.input_ids,
                             global_attention_tokens=new_token_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(ens_model_n)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.to(device)

    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm.tqdm(range(num_training_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    best_dict = model.state_dict()
    best_val_score, tol, determining_score_idx = 0, 0, 0
    for epoch in range(args.num_epochs):
        running_loss1, running_loss2 = [], []
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            running_loss1.append(loss.item())
            progress_bar.set_postfix({"Train Loss": np.mean(running_loss1)})
            torch.cuda.empty_cache()

        scores = test_loop(model, tokenizer, val_loader, device, task_name=args.task_name)
        acc_mean = scores[determining_score_idx]
        if acc_mean > best_val_score:
            best_val_score = acc_mean
            best_dict = model.state_dict()
            tol = 0
        else:
            tol += 1

        if tol >= 3:
            print("early stopping...")
            break

    print("Testing model performance")
    model.load_state_dict(best_dict)
    test_scores = test_loop(model, tokenizer, test_loader, device, mode="Test", task_name=args.task_name)
    score_str = f"Combinations {args.model_ids} \t scores:{test_scores}\n"
    comb_code = "".join(map(lambda x: str(x), args.model_ids))
    scores_path = os.path.join("results", f"scores_{args.task_name}_{comb_code}.txt")
    with open(scores_path, "a") as f:
        f.write(score_str)

    print("Saving model...")
    model_save_path = os.path.join("results", "ens_models",
                                   f"best_result_{args.task_name}_{comb_code}")
    model.save_pretrained(model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='focal diversity pruning')
    parser.add_argument('--task_name', default="search_qa", type=str, choices=["search_qa", "gsm8k", "mmlu"])
    parser.add_argument('--model_ids', default="013", type=str, required=True)
    parser.add_argument('--num_k', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--train_percentage', default=1.0, type=float)
    parser.add_argument("--skip_model_outputs", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()
    main(args)
