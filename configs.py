DATA_DIR = "data"
RESULT_DIR = "results"


model_names_dict = {
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
    "mmlu": {
        0: "Mixtral-8x7B-Instruct-v0.1",
        1: "gemma-7b-it",
        2: "Llama-2-13b-chat-hf",
        3: "Llama-2-70b-chat-hf",
        4: "Mistral-7B-Instruct-v0.1"
    },
    "search_qa": {
        0: "Mixtral-8x7B-Instruct-v0.1",
        1: "gemma-7b-it",
        2: "Llama-2-13b-chat-hf",
        3: "Llama-2-70b-chat-hf",
        4: "Mistral-7B-Instruct-v0.1"
    }
}
