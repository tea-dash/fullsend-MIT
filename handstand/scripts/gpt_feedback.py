import json
import os
from typing import Any, Dict, List

from lib.io_utils import load_json, save_json


PROMPT_TEMPLATE = """You are an expert handstand coach and biomechanist.
Given the baseline metrics (mean, std) and the user's metrics and z-scores below,
identify the top 5 limiting factors in the user's static freestanding floor handstand.

For each factor:
- Explain the impact on balance/line/efficiency
- State when it is most evident during the hold
- Provide 2–3 concrete corrections and one targeted drill

Be concise, technical, and specific. Use biomechanics terms where appropriate. Output in markdown.

BASELINE_KEYS:
{baseline_keys}

USER_METRICS (deg / unit):
{user_metrics}

ZSCORES (positive means user's value > baseline mean):
{zscores}
"""


def render_prompt(payload: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        baseline_keys=json.dumps(payload.get("baseline_keys", []), indent=2),
        user_metrics=json.dumps(payload.get("user_metrics", {}), indent=2),
        zscores=json.dumps(payload.get("zscores", {}), indent=2),
    )


def call_openai_markdown(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-5")
    if not api_key:
        return (
            "OPENAI_API_KEY not set. Skipping API call.\n\n"
            "Prompt would have been:\n\n" + prompt
        )
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return f"openai package not installed properly ({e}).\n\nPrompt:\n{prompt}"
    client = OpenAI(api_key=api_key)
    try:
        if model.startswith("gpt-5"):
            # Use Responses API for latest models; omit temperature
            resp = client.responses.create(
                model=model,
                input=prompt,
            )
            text = resp.output_text or ""
        else:
            # Chat Completions for established chat models
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful expert handstand coach."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
        return text
    except Exception as e:
        return f"OpenAI API error: {e}\n\nPrompt:\n{prompt}"


def main():
    report_path = os.path.join("data", "user", "metrics", "user_vs_baseline.json")
    if not os.path.exists(report_path):
        raise RuntimeError("user_vs_baseline.json not found. Run analyze_user_clip.py first.")
    payload = load_json(report_path)
    prompt = render_prompt(payload)
    md = call_openai_markdown(prompt)
    out_md = os.path.join("data", "user", "metrics", "gpt_feedback.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"GPT feedback saved to {out_md}")


if __name__ == "__main__":
    main()


