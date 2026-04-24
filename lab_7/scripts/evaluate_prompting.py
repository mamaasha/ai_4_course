"""Расчет метрик"""


import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def read_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


@dataclass
class EvalSample:
    """анкета и метка"""
    profile: str
    label: int


def load_dataset(path: str) -> list[EvalSample]:
    rows: list[EvalSample] = []
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(EvalSample(profile=row["profile"], label=int(row["label"])))
    return rows


def parse_verdict(answer: str) -> int:
    """Достает вердикт 0 или 1 из ответа"""
    answer = answer.strip()
    try:
        payload = json.loads(answer)
        verdict = int(payload.get("verdict", 0))
        return 1 if verdict == 1 else 0
    except Exception:
        match = re.search(r"\b[01]\b", answer)
        if match:
            return int(match.group(0))
    return 0


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """accuracy, precision, recall и F1"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def build_prompts() -> dict[str, str | None]:
    """Возвращает промпты"""
    return {
        "zero-shot": None,
        "cot": read_prompt("eval_cot_system.md"),
        "few-shot": read_prompt("eval_few_shot_system.md"),
        "cot+few-shot": read_prompt("eval_cot_few_shot_system.md"),
    }


def ask_llm(
    base_url: str,
    provider: str,
    profile: str,
    system_prompt: str | None,
    timeout: int,
    retries: int,
) -> int:
    """Отправляет запрос в LLM"""
    user_prompt = read_prompt("eval_user_prompt.md").format(profile=profile)
    payload = {"provider": provider, "user_prompt": user_prompt}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    for attempt in range(retries + 1):
        try:
            response = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
            response.raise_for_status()
            answer = response.json()["answer"]
            return parse_verdict(answer)
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(1)
    return 0


def evaluate(base_url: str, provider: str, dataset_path: str, timeout: int, retries: int) -> dict:
    """Полный запуск"""
    dataset = load_dataset(dataset_path)
    prompts = build_prompts()
    report = {}
    workers = 8

    for technique, system_prompt in prompts.items():
        y_true = [sample.label for sample in dataset]
        y_pred = [0] * len(dataset)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, sample in enumerate(dataset):
                future = pool.submit(
                    ask_llm,
                    base_url=base_url,
                    provider=provider,
                    profile=sample.profile,
                    system_prompt=system_prompt,
                    timeout=timeout,
                    retries=retries,
                )
                futures[future] = i
            for future in as_completed(futures):
                idx = futures[future]
                y_pred[idx] = future.result()
        report[technique] = compute_metrics(y_true=y_true, y_pred=y_pred)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--dataset", default="ai_4_course/lab_6/data/petfinder_small.csv")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    report = evaluate(
        base_url=args.base_url,
        provider=args.provider,
        dataset_path=args.dataset,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
