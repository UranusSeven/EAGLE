import logging
import os

import fire
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


def build_instruct_chat_50k(dataset, start, end, tokenizer, n_proc=4):
    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        from fastchat.model.model_adapter import get_conversation_template

        ret = {"conversation": [], "input_ids": [], "loss_mask": []}
        for i in range(len(batch["input"])):
            # Construct a single round conversation.
            conv = get_conversation_template("qwen1.5-7b-chat")
            inst: str = batch["input"][i][0]
            resp: str = batch["output"][i][0]
            conv.append_message(conv.roles[0], inst)
            conv.append_message(conv.roles[1], resp)
            conversation = conv.get_prompt()
            # Ignore the system prompt and user instruction.
            masked_part = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"
            )
            if not conversation.startswith(masked_part):
                continue

            input_ids: list[int] = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
            masked_input_ids: list[int] = tokenizer(
                masked_part,
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]

            if len(input_ids) > 4096:
                continue

            loss_mask = torch.ones_like(input_ids)
            loss_mask[: len(masked_input_ids)] = 0

            ret["conversation"].append(conversation)
            # [bs, seqlen]
            ret["input_ids"].append(input_ids[None, :])
            # [bs, seqlen]
            ret["loss_mask"].append(loss_mask[None, :])

        return ret

    dataset = dataset.select(range(start, end))
    dataset = dataset.map(
        tokenize, batched=True, num_proc=n_proc, remove_columns=dataset.column_names
    )
    return dataset


def tokenize(
    tokenizer,
    system_message: str,
    messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    """
    messages: [
        {
            "role": "user",
            "content": "instruction..."
        },
        {
            "role": "assistant",
            "content": "response..."
        }
        ...
    ]
    """
    input_ids = []
    loss_mask = []
    
    roles = {
        "user": "<|im_start|>user",
        "assistant": "<|im_start|>assistant"
    }
    sep = "<|im_end|>"
    sys_prompt = f"<|im_start|>system\n{system_message}" + sep + "\n"

    sys_input_ids = tokenizer(sys_prompt).input_ids
    input_ids += sys_input_ids
    loss_mask += [0] * len(sys_input_ids)

    for message in messages:
        role = message["role"]
        content = message["content"]
        
        role_ids = tokenizer(
            roles[message["role"]] + "\n", add_special_tokens=False
        ).input_ids
        content_ids = tokenizer(
            content, add_special_tokens=False
        ).input_ids
        sep_ids = tokenizer(
            sep + "\n", add_special_tokens=False
        ).input_ids
        input_ids += role_ids + content_ids + sep_ids
        loss_mask += [0] * len(role_ids)
        if role == "user":
            loss_mask += [0] * len(content_ids)
        elif role == "assistant":
            loss_mask += [1] * len(content_ids)
        else:
            raise RuntimeError(f"Unexpected role: {role}")
        loss_mask += [0] * len(sep_ids)
    
    # assertion
    from fastchat.model.model_adapter import get_conversation_template
    conv = get_conversation_template("qwen1.5-7b-chat")
    conv.system_message = system_message
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            conv.append_message(conv.roles[0], content)
        elif role == "assistant":
            conv.append_message(conv.roles[1], content)
        else:
            raise RuntimeError(f"Unexpected role: {role}")
            
    # assert input_ids == tokenizer(conv.get_prompt()).input_ids, f"expected:\n{tokenizer(conv.get_prompt()).input_ids}\nactual:\n{input_ids}"
    if input_ids != tokenizer(conv.get_prompt()).input_ids:
        return None, None, None
    assert len(input_ids) == len(loss_mask)
    return conv.get_prompt(), input_ids, loss_mask


def build_sharegpt_chinese_english_90k(dataset, start, end, tokenizer, n_proc=4):
    def to_messages(conversation):
        messages = []
        for t in conversation:
            messages.append({
                "role": "user",
                "content": t["human"]
            })
            messages.append({
                "role": "assistant",
                "content": t["assistant"]
            })
        return messages
    
    def process_batch(batch: dict[str, list]) -> dict[str, list]:
        from fastchat.model.model_adapter import get_conversation_template

        ret = {"conversation": [], "input_ids": [], "loss_mask": []}
        for i in range(len(batch["conversation_id"])):
            messages = to_messages(batch["conversation"][i])
            conversation, input_ids, loss_mask = tokenize(
                tokenizer=tokenizer,
                system_message="You are a helpful assistant.",
                messages=messages
            )
            
            if conversation is not None:
                ret["conversation"].append(conversation)
                # [bs, seqlen]
                ret["input_ids"].append([input_ids])
                # [bs, seqlen]
                ret["loss_mask"].append([loss_mask])

        return ret

    dataset = dataset.select(range(start, end))
    dataset = dataset.map(
        process_batch, batched=True, num_proc=n_proc, remove_columns=dataset.column_names
    )
    return dataset


@torch.no_grad()
def get_hidden_states(model, example):
    # [bs, seqlen]
    input_ids = example["input_ids"]
    out = model(
        torch.as_tensor(input_ids, dtype=torch.int).cuda(),
        output_hidden_states=True
    )
    hidden_states = out.hidden_states[-1]
    return {
        # [seqlen,]
        "input_ids": torch.as_tensor(input_ids, dtype=torch.int)[0],
        # [seqlen, hidden_size]
        "hidden_state": hidden_states.cpu()[0],
        # [seqlen,]
        "loss_mask": torch.as_tensor(example["loss_mask"], dtype=torch.int)[0],
    }


def main(
    output_dir: str,
    model_path: str,
    dataset_path: str,
    start: int = 0,
    end: int = 10,
    n_proc: int = 1,
    n_gpus: int = 1,
):
    logger.info(f"Loading dataset from `{dataset_path}`.")
    ds = load_dataset("json", data_files=dataset_path)["train"]
    logger.info(f"The dataset has been loaded.")

    logger.info(f"Tokenizing the dataset.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # tokenized_ds = build_instruct_chat_50k(
    #     dataset=ds, start=start, end=end, tokenizer=tokenizer, n_proc=n_proc
    # )
    tokenized_ds = build_sharegpt_chinese_english_90k(
        dataset=ds, start=start, end=end, tokenizer=tokenizer, n_proc=n_proc
    )
    logger.info(f"The dataset has been tokenized.")

    logger.info(f"Loading model from {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    device_map = {
        "model.embed_tokens": 0,
        "lm_head": n_gpus - 1,
        "model.norm": n_gpus - 1,
    }
    for i in range(config.num_hidden_layers):
        device_map[f"model.layers.{i}"] = i * n_gpus // config.num_hidden_layers
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device_map
    ).cuda()
    model.eval()
    logger.info("The model has been loaded.")

    logger.info(f"Generating trainning data, output dir: {output_dir}")
    assert os.path.isdir(output_dir), f"{output_dir} must be a directory"
    for i, example in tqdm(enumerate(tokenized_ds)):
        output_path = os.path.join(output_dir, f"example_{i}.ckpt")
        if os.path.exists(output_path):
            logger.info(f"Skip existed example {i}.")
        torch.save(get_hidden_states(model=model, example=example), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
