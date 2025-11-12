import argparse
import copy
import json
import os
from collections import defaultdict

import uvloop
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
import tiktoken

from utils import extract_solution, update_answer
from utils.aio import async_main, close_async_client
from utils.envs import DATAROOT, override_endpoint

DOCS = None
ARGS = None


def string_match_all(pred, ref):
    return sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)


def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {"sub_em": 0, "total_num": 0}
    for pred, gold in zip(predictions, goldens):
        metrics["sub_em"] += string_match_all(pred, gold)
    metrics["total_num"] = len(goldens)
    for key in list(metrics.keys()):
        if key == "total_num":
            continue
        metrics[key] = round(metrics[key] / metrics["total_num"], 2)
    return metrics


def calc_qa_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {"f1": 0, "prec": 0, "recall": 0, "em": 0, "sub_em": 0, "total_num": 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for key in list(metrics.keys()):
        if key == "total_num":
            continue
        metrics[key] = round(metrics[key] / metrics["total_num"], 2)
    return metrics


def read_squad(file_path):
    with open(file_path) as f:
        data = json.load(f)
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {context: idx for idx, context in enumerate(total_docs)}
    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [idx for idx in more_docs if idx != total_docs_dict[p["context"]]],
                        }
                    )
    return total_qas, total_docs


def read_hotpotqa(file_path):
    with open(file_path) as f:
        data = json.load(f)
    total_docs = [f"{title}\n{''.join(paragraph)}" for d in data for title, paragraph in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {context: idx for idx, context in enumerate(total_docs)}
    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [total_docs_dict[f"{title}\n{''.join(paragraph)}"] for title, paragraph in d["context"]],
            }
        )
    return total_qas, total_docs


def set_context(item):
    global DOCS, ARGS
    if DOCS is None:
        if ARGS.split == "qa_1":
            _, DOCS = read_squad("../memory_data/squad.json")
        elif ARGS.split == "qa_2":
            _, DOCS = read_hotpotqa("../memory_data/hotpotqa_dev.json")
        else:
            raise ValueError(f"Unsupported split for context lifting: {ARGS.split}")
    all_docs = [DOCS[idx] for idx in item["context"]]
    document_prompt = "Document {i}:\n{document}"
    context = "\n\n".join([document_prompt.format(i=i + 1, document=doc) for i, doc in enumerate(all_docs)])
    item["context"] = context
    return item


def build_local_url(args):
    if args.url:
        url = args.url.rstrip("/")
        return url if url.endswith("/v1") else f"{url}/v1"
    scheme = args.scheme
    host = args.local_host
    port = args.local_port
    return f"{scheme}://{host}:{port}/v1"


def configure_local_endpoint(args):
    local_url = build_local_url(args)
    override_endpoint(url=local_url, api_key=args.api_key)


def resolve_async_client(args):
    if args.api == "recurrent":
        from utils.recurrent import async_query_llm  # noqa: F401
        from utils import extract_answer  # noqa: F401
        return ("recurrent", async_query_llm, extract_answer)
    if args.api == "recurrent-boxed":
        from utils.recurrent_boxed import async_query_llm  # noqa: F401
        from utils import extract_boxed_answer as extract_answer  # noqa: F401
        return ("recurrent-boxed", async_query_llm, extract_answer)
    if args.api == "boxed":
        from utils.boxed import async_query_llm  # noqa: F401
        from utils import extract_boxed_answer as extract_answer  # noqa: F401
        return ("boxed", async_query_llm, extract_answer)
    if args.api == "openai":
        from utils.openai_api import async_query_llm  # noqa: F401
        from utils import extract_answer  # noqa: F401
        return ("openai", async_query_llm, extract_answer)
    raise ValueError(f"Unsupported API type: {args.api}")


def choose_tokenizer(args):
    model = args.model
    if any(key in model for key in ["gpt", "o1", "o3", "o4", "gemini", "claude"]):
        return tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    return AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)


def evaluate_dataset(data, args, out_file):
    tokenizer = choose_tokenizer(args)
    api_tag, async_query_llm, extract_answer = resolve_async_client(args)
    coros = [async_query_llm(item, args.model, tokenizer, temperature=args.temperature, top_p=args.top_p) for item in data]
    outputs = uvloop.run(async_main(coros, args.n_proc))
    uvloop.run(async_main([close_async_client()]))
    scores = defaultdict(list)
    mode = "w" if args.force else "a"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, mode, encoding="utf-8") as fout:
        for idx, (output, item) in enumerate(zip(outputs, data)):
            if output == "":
                continue
            response = output.strip()
            pred_raw, _ = extract_solution(response)
            item["response"] = response
            item["answer"] = item.pop("outputs")
            item["pred"] = extract_answer(pred_raw) if pred_raw else extract_answer(response)
            if "qa" in args.split:
                if item["pred"]:
                    metrics = calc_qa_metrics([item["pred"]], [item["answer"][0]])
                else:
                    metrics = {"f1": 0, "prec": 0, "recall": 0, "em": 0, "sub_em": 0, "total_num": 0}
                item["judge_sub_em"] = metrics["sub_em"]
                item["judge_em"] = metrics["em"]
                item["judge_f1"] = metrics["f1"]
                scores["em"].append(item["judge_em"])
                scores["f1"].append(item["judge_f1"])
                scores["sub_em"].append(item["judge_sub_em"])
            else:
                item["judge_sub_em"] = calc_metrics([item["pred"]], [item["answer"]])["sub_em"] if item["pred"] else 0
                scores["sub_em"].append(item["judge_sub_em"])
            item.pop("context")
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            if idx == 0:
                print("=" * 40 + "New Item Start" + "=" * 40)
                print(item["response"])
                print("-" * 80)
                print(item["pred"])
                print("-" * 80)
                print(item["answer"])
                print("-" * 80)
                print(item["judge_sub_em"])
                print("=" * 40 + "New Item End" + "=" * 40)
    print(f"ruler_general_local [{args.length}] via {api_tag}")
    for key, values in scores.items():
        print(f"{key}: {round(sum(values) * 100 / len(values), 2)}")
    print(f"Total: {len(data)}")


def main():
    global DOCS, ARGS
    DOCS = None
    ARGS = parse_args()
    configure_local_endpoint(ARGS)
    print(ARGS)
    out_file = os.path.join(ARGS.save_dir, ARGS.save_file + ".jsonl")
    dataset = concatenate_datasets([load_dataset("json", data_files=f"{DATAROOT}/eval_{ARGS.split}_{ARGS.length}.json", split="train")])
    if isinstance(dataset[0]["context"], list):
        dataset = [set_context(item) for item in dataset]
    dataset = [copy.deepcopy(item) for _ in range(ARGS.sampling) for item in dataset]
    for idx, item in enumerate(dataset):
        item["_id"] = idx
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as fin:
            has_data = {json.loads(line)["_id"]: 0 for line in fin}
    data = []
    for item in dataset:
        if item["_id"] not in has_data or ARGS.force:
            data.append(item)
    if not data:
        print("无需评测，结果文件已存在且未指定 --force。")
        return
    evaluate_dataset(data, ARGS, out_file)


def parse_args():
    parser = argparse.ArgumentParser(description="RULER General 评测脚本（本地模型版）")
    parser.add_argument("--split", type=str, default="niah_single_1", choices=[
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "cwe",
        "fwe",
        "qa_1",
        "qa_2",
    ])
    parser.add_argument("--length", type=int, default=8192, choices=[
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        1048576 * 2,
        1048576 * 4,
        10000000,
    ])
    parser.add_argument("--save_dir", "-s", type=str, default="results/ruler_general_local")
    parser.add_argument("--save_file", "-f", type=str, default="local-model-recurrent")
    parser.add_argument("--model", "-m", type=str, default="local-model")
    parser.add_argument("--tokenizer", "-t", type=str, required=True)
    parser.add_argument("--n_proc", "-n", type=int, default=64)
    parser.add_argument("--api", "-a", type=str, default="recurrent", choices=["recurrent", "recurrent-boxed", "boxed", "openai"])
    parser.add_argument("--sampling", "-p", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="覆写已有结果")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--scheme", type=str, default="http", choices=["http", "https"])
    parser.add_argument("--local-host", dest="local_host", type=str, default="127.0.0.1")
    parser.add_argument("--local-port", dest="local_port", type=int, default=8000)
    parser.add_argument("--api-key", dest="api_key", type=str, default="local-test-key")
    parser.add_argument("--url", type=str, default=None, help="若已启动 OpenAI 兼容服务，可直接指定完整 URL（可含 /v1）")
    return parser.parse_args()


if __name__ == "__main__":
    main()
