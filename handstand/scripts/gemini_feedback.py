import argparse
import json
import os
from typing import Any, Dict

import google.generativeai as genai  # type: ignore

from lib.io_utils import load_json


PROMPT_TEMPLATE = """You are an expert handstand coach. Review the baseline metrics (mean, std) and the user's metrics with z-scores. Identify the top technique issues (limit to 5), explain their impact on balance/line, describe when in the hold they appear, and give 2-3 precise corrections plus one drill each. Keep the response concise and technical.

BASELINE_KEYS:
{baseline_keys}

USER_METRICS:
{user_metrics}

ZSCORES:
{zscores}
"""


def render_prompt(payload: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        baseline_keys=json.dumps(payload.get("baseline_keys", []), indent=2),
        user_metrics=json.dumps(payload.get("user_metrics", {}), indent=2),
        zscores=json.dumps(payload.get("zscores", {}), indent=2),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate feedback using Google Gemini LLM.")
    parser.add_argument(
        "--analysis-json",
        default="data/user/metrics/user_vs_baseline.json",
        help="Path to analyze_user_clip output JSON.",
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-pro",
        help="Gemini model name (e.g., gemini-1.5-pro).",
    )
    parser.add_argument(
        "--out",
        default="data/user/metrics/gemini_feedback.md",
        help="Path to save markdown feedback.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)

    payload = load_json(args.analysis_json)
    prompt = render_prompt(payload)

    model = genai.GenerativeModel(args.model)
    response = model.generate_content(prompt)
    text = response.text or ""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Gemini feedback saved to {args.out}")


if __name__ == "__main__":
    main()


