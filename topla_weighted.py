import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import RESULT_DIR
from helper import (load_mmlu_prob_and_label, load_mmlu_multi_run,
                    load_gsm8k_prob_and_label, get_mean_acc_models)


model_mapper_dict = {
    "gsm8k": {
        0: "Llama-2-7b-chat-hf",
        1: "Llama-2-13b-chat-hf",
        2: "Mixtral-8x7B-Instruct-v0.1",
        3: "gemma-7b",
        4: "gemma-2b",
        5: "Llama-2-70b-chat-hf",
        6: "Mistral-7B-Instruct-v0.2",
        7: "phi-2"
    },
    "mmlu_hf": {
        0: "Llama-2-7b-hf",
        1: "Llama-2-13b-hf",
        2: "Mixtral-8x7B-v0.1",
        3: "gemma-7b",
        4: "gemma-2b",
        5: "Llama-2-70b-hf",
        6: "Mistral-7B-Instruct-v0.2",
        7: "phi-2",
    },
    "mmlu_togetherai": {
        0: "Llama-2-7b-chat-hf",
        1: "Llama-2-13b-chat-hf",
        2: "Mixtral-8x7B-Instruct-v0.1",
        3: "gemma-7b-it",
        4: "gemma-2b-it",
        5: "Llama-2-70b-chat-hf",
        6: "Mistral-7B-Instruct-v0.1"
    }
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_dim[1], output_dim),
            nn.Sigmoid()
        )
        self.net.apply(self.init_weights)

    def forward(self, x):
        out = self.net(x)
        out = torch.softmax(out, dim=-1)
        return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


def train_ensemble(model_names, train_loader, val_loader, novel_loader, n_epochs, save_dir, space_size, verbose=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = MLP(len(model_names) * space_size, [100, 100], space_size)
    model = model.to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc, tol = (0, 0)
    for epoch in range(n_epochs):
        avg_loss = []
        for i, batch_data in enumerate(train_loader):
            in_x = batch_data[:, :-1].to("cuda").float()
            label = batch_data[:, -1].type(torch.long).to("cuda")

            optimizer.zero_grad()
            out = model(in_x)
            loss = loss_fn(out, label)

            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        if epoch % 10 == 0 and verbose:
            run_loss = np.mean(avg_loss)
            print(f'Epoch {epoch} | Loss {run_loss:.4f}')

        if val_loader:
            acc_mean = test_loop(model, val_loader)

            if acc_mean > best_val_acc:
                outfile = os.path.join(save_dir, f'best_model.tar')
                torch.save({'epoch': epoch,
                            'state': model.state_dict(),
                            "accuracy": acc_mean}, outfile)
                best_val_acc = acc_mean
                tol = 0
            else:
                tol += 1

            if tol > 200:
                print("No improvement in 200 epochs, breaking")
                break

    if val_loader:
        best_dict = torch.load(f"{save_dir}/best_model.tar")
        model.load_state_dict(best_dict["state"])

    model.eval()
    acc_mean = test_loop(model, novel_loader)
    print(f'Novel Acc = {acc_mean:.4f}')
    exp_result = dict(val_acc=best_dict["accuracy"],
                      val_conf=best_dict["confidence"],
                      test_acc=acc_mean,
                      state=model.state_dict(),
                      model_names=model_names)

    return exp_result


def test_loop(model, data_loader, ret_logit=False, device="cuda"):
    assert device in ["cuda", "cpu"]
    acc_all = []
    logits = []
    labels = []
    for i, batch_data in enumerate(data_loader):
        in_x = batch_data[:, :-1].to(device).float()
        scores = model(in_x)
        label = batch_data[:, -1].numpy()

        scores = scores.detach().cpu().numpy()
        in_x = in_x.detach().cpu().numpy()
        pred = np.argmax(scores, axis=1)
        corrects = np.sum(pred == label)
        acc_all.append(corrects / len(label) * 100)
        if ret_logit:
            logits.append(np.concatenate([in_x, scores], axis=1))
            labels.append(label)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)

    if ret_logit:
        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        return acc_mean, logits, labels
    else:
        return acc_mean


def run(args):
    np.random.seed(args.seed)
    model_names = [model_mapper_dict[args.task_name][int(i)] for i in args.model_ids]
    space_size = 30 if args.task_name == 'gsm8k' else 4
    pred_dir = os.path.join(RESULT_DIR, args.task_name)

    if args.task_name == "gsm8k":
        get_mean_acc_models(pred_dir, model_names, dataset_name="test")
        train_data = load_gsm8k_prob_and_label(pred_dir, model_names,
                                               num_runs=args.num_k,
                                               space_size=space_size,
                                               drop_non_exists=True,
                                               dataset_name="train")
        test_data = load_gsm8k_prob_and_label(pred_dir, model_names,
                                              num_runs=args.num_k,
                                              num_samples=None,
                                              space_size=space_size,
                                              drop_non_exists=False,
                                              dataset_name="test")
        random_idx = np.random.permutation(range(len(train_data)))
        train_data = train_data[random_idx]

        val_size = int(len(train_data) * 0.25)
        train_size = len(train_data) - val_size

        novel_data = test_data[:, :-1]
        novel_label = test_data[:, -1]
        for novel_arr in np.split(novel_data, len(model_names), axis=1):
            pred = novel_arr.argmax(axis=1)
            print(np.mean(novel_label == pred))

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(train_data[train_size:], batch_size=args.batch_size, shuffle=True)
        novel_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    else:
        if args.task_name == "mmlu_hf":
            data = load_mmlu_multi_run(model_names, "test")
        else:
            data = load_mmlu_prob_and_label(model_names)
        data = np.concatenate(data)

        rand_idx = np.random.permutation(len(data))
        data = data[rand_idx]
        ds_len = len(data)
        train_size = int(ds_len * 0.7)
        val_size = int(ds_len * 0.1)
        split = {"train": data[:train_size],
                 "val": data[train_size:train_size + val_size],
                 "test": data[train_size + val_size:]}

        train_loader = DataLoader(split["train"], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(split["val"], batch_size=args.batch_size, shuffle=True)
        novel_loader = DataLoader(split["test"], batch_size=args.batch_size, shuffle=True)

    train_ensemble(model_names, train_loader, val_loader, novel_loader,
                   n_epochs=300, save_dir="results/ensemble",
                   space_size=space_size, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='focal diversity pruning')
    parser.add_argument('--task_name', default="gsm8k", type=str,
                        choices=["gsm8k", "mmlu_hf", "mmlu_togetherai"])
    parser.add_argument('--model_ids', default="237", type=str, required=True)
    parser.add_argument('--num_k', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--train_percentage', default=1.0, type=float)
    parser.add_argument('--save_freq', default=100, type=float)
    parser.add_argument('--seed', default=9, type=int)
    arguments = parser.parse_args()
    run(arguments)
