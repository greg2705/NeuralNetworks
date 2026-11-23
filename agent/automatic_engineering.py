"""Query MOSS-Transcribe-Diarize served by vLLM (/v1/audio/transcriptions).

Usage:
    python transcribe_moss.py meeting.wav
    python transcribe_moss.py meeting.wav --language fr --expected-speakers 3
"""
import argparse
import json

import requests

API_BASE_URL = "http://localhost:8000/v1"  # <- set your vLLM address here
MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"  # or your --served-model-name


def transcribe(path, language=None, prompt=None, api_base_url=API_BASE_URL):
    data = {"model": MODEL, "response_format": "diarized_json", "temperature": "0"}
    if language:
        data["language"] = language  # standard vLLM field; may be ignored by this model
    if prompt:
        data["prompt"] = prompt  # free-text hint, e.g. hotwords (experimental)
    with open(path, "rb") as f:
        r = requests.post(
            f"{api_base_url}/audio/transcriptions",
            files={"file": f},
            data=data,
            timeout=3600,
        )
    r.raise_for_status()
    return r.json()


def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--language", help="ISO code hint, e.g. fr, en")
    p.add_argument("--prompt", help="optional text hint (hotwords, context)")
    p.add_argument("--expected-speakers", type=int,
                   help="only used to warn if the model finds a different count")
    p.add_argument("--api-base-url", default=API_BASE_URL)
    p.add_argument("--json-out", help="save the raw response to this file")
    a = p.parse_args()

    res = transcribe(a.audio, a.language, a.prompt, a.api_base_url)
    segments = res.get("segments", [])

    for s in segments:
        print(f"[{fmt(s['start'])} -> {fmt(s['end'])}] {s['speaker']}: {s['text']}")

    speakers = sorted({s["speaker"] for s in segments})
    print(f"\n{len(speakers)} speaker(s) detected: {', '.join(speakers)}")
    if a.expected_speakers and a.expected_speakers != len(speakers):
        print(f"Warning: expected {a.expected_speakers}, model found {len(speakers)}")

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
