import asyncio
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# Works with an already-configured OpenAI or AsyncOpenAI client
# expected interface:
#   await client.chat.completions.create(..., stream=True)

@dataclass
class RequestResult:
    ok: bool
    error: Optional[str]
    started_at: float
    ended_at: float
    latency_s: float
    ttft_s: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    output_text_len: int

    @property
    def output_tps(self) -> Optional[float]:
        if self.output_tokens is None or self.latency_s <= 0:
            return None
        return self.output_tokens / self.latency_s


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


async def run_one_request(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    request_kwargs: Optional[Dict[str, Any]] = None,
) -> RequestResult:
    request_kwargs = request_kwargs or {}

    started_at = time.perf_counter()
    first_token_at = None
    ended_at = None

    output_chunks = []
    input_tokens = None
    output_tokens = None
    total_tokens = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **request_kwargs,
        )

        async for chunk in stream:
            now = time.perf_counter()

            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None

                if content:
                    if first_token_at is None:
                        first_token_at = now
                    output_chunks.append(content)

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_tokens", input_tokens)
                output_tokens = getattr(usage, "completion_tokens", output_tokens)
                total_tokens = getattr(usage, "total_tokens", total_tokens)

        ended_at = time.perf_counter()

        return RequestResult(
            ok=True,
            error=None,
            started_at=started_at,
            ended_at=ended_at,
            latency_s=ended_at - started_at,
            ttft_s=(first_token_at - started_at) if first_token_at else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            output_text_len=len("".join(output_chunks)),
        )

    except Exception as e:
        ended_at = time.perf_counter()
        return RequestResult(
            ok=False,
            error=f"{type(e).__name__}: {e}",
            started_at=started_at,
            ended_at=ended_at,
            latency_s=ended_at - started_at,
            ttft_s=(first_token_at - started_at) if first_token_at else None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            output_text_len=0,
        )


async def run_benchmark_round(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    concurrency: int,
    num_requests: int,
    request_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request_kwargs = request_kwargs or {}

    sem = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []

    async def worker():
        async with sem:
            return await run_one_request(
                client=client,
                model=model,
                messages=messages,
                request_kwargs=request_kwargs,
            )

    started_at = time.perf_counter()
    tasks = [asyncio.create_task(worker()) for _ in range(num_requests)]
    for t in asyncio.as_completed(tasks):
        results.append(await t)
    ended_at = time.perf_counter()

    wall_time_s = ended_at - started_at
    oks = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]

    latencies = [r.latency_s for r in oks]
    ttfts = [r.ttft_s for r in oks if r.ttft_s is not None]
    output_tps_per_req = [r.output_tps for r in oks if r.output_tps is not None]
    output_tokens_sum = sum(r.output_tokens for r in oks if r.output_tokens is not None)
    input_tokens_sum = sum(r.input_tokens for r in oks if r.input_tokens is not None)

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successes": len(oks),
        "failures": len(errs),
        "wall_time_s": wall_time_s,
        "requests_per_s": (len(results) / wall_time_s) if wall_time_s > 0 else None,
        "successful_requests_per_s": (len(oks) / wall_time_s) if wall_time_s > 0 else None,
        "total_input_tokens": input_tokens_sum if input_tokens_sum > 0 else None,
        "total_output_tokens": output_tokens_sum if output_tokens_sum > 0 else None,
        "aggregate_output_tokens_per_s": (output_tokens_sum / wall_time_s)
        if wall_time_s > 0 and output_tokens_sum > 0
        else None,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "per_request_output_tps_mean": statistics.mean(output_tps_per_req)
        if output_tps_per_req
        else None,
        "sample_errors": [r.error for r in errs[:5]],
        "raw_results": [asdict(r) for r in results],
    }


async def benchmark_vllm_openai_client(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    concurrency_levels: List[int] = (1, 2, 4, 8, 16, 32),
    num_requests_per_level: int = 50,
    max_tokens: int = 128,
    temperature: float = 0.0,
    extra_request_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Runs the same prompt at multiple concurrency levels.

    Parameters
    ----------
    client:
        An AsyncOpenAI-compatible client already configured for your vLLM endpoint.
    model:
        Model name served by vLLM.
    messages:
        OpenAI chat-format messages.
    concurrency_levels:
        Concurrent in-flight request counts to test.
    num_requests_per_level:
        Total requests to send per concurrency level.
    max_tokens:
        Max generation length.
    temperature:
        Sampling temperature.
    extra_request_kwargs:
        Any extra params supported by your vLLM server.

    Returns
    -------
    List of round summary dicts.
    """
    request_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        **(extra_request_kwargs or {}),
    }

    all_rounds = []
    for concurrency in concurrency_levels:
        summary = await run_benchmark_round(
            client=client,
            model=model,
            messages=messages,
            concurrency=concurrency,
            num_requests=num_requests_per_level,
            request_kwargs=request_kwargs,
        )
        all_rounds.append(summary)

    return all_rounds


def print_benchmark_table(results: List[Dict[str, Any]]) -> None:
    headers = [
        "conc",
        "ok/total",
        "req/s",
        "out tok/s",
        "lat p50",
        "lat p95",
        "ttft p50",
        "ttft p95",
    ]
    print(
        f"{headers[0]:>5}  {headers[1]:>10}  {headers[2]:>10}  {headers[3]:>12}  "
        f"{headers[4]:>9}  {headers[5]:>9}  {headers[6]:>9}  {headers[7]:>9}"
    )

    for r in results:
        print(
            f"{r['concurrency']:>5}  "
            f"{f'{r['successes']}/{r['num_requests']}':>10}  "
            f"{(f'{r['successful_requests_per_s']:.2f}' if r['successful_requests_per_s'] is not None else '-'):>10}  "
            f"{(f'{r['aggregate_output_tokens_per_s']:.2f}' if r['aggregate_output_tokens_per_s'] is not None else '-'):>12}  "
            f"{(f'{r['latency_p50_s']:.3f}' if r['latency_p50_s'] is not None else '-'):>9}  "
            f"{(f'{r['latency_p95_s']:.3f}' if r['latency_p95_s'] is not None else '-'):>9}  "
            f"{(f'{r['ttft_p50_s']:.3f}' if r['ttft_p50_s'] is not None else '-'):>9}  "
            f"{(f'{r['ttft_p95_s']:.3f}' if r['ttft_p95_s'] is not None else '-'):>9}"
        )


# Example usage:
#
# import asyncio
# from openai import AsyncOpenAI
#
# client = AsyncOpenAI(
#     base_url="http://localhost:8000/v1",
#     api_key="dummy",  # vLLM often accepts any non-empty string if auth is disabled
# )
#
# messages = [
#     {"role": "system", "content": "You are a concise assistant."},
#     {"role": "user", "content": "Write a 200-word explanation of attention in transformers."},
# ]
#
# async def main():
#     results = await benchmark_vllm_openai_client(
#         client=client,
#         model="your-served-model-name",
#         messages=messages,
#         concurrency_levels=[1, 2, 4, 8, 16, 32],
#         num_requests_per_level=20,
#         max_tokens=200,
#         temperature=0.0,
#         extra_request_kwargs={
#             # if your server supports it, this can expose usage in streamed chunks
#             # "stream_options": {"include_usage": True},
#         },
#     )
#     print_benchmark_table(results)
#
#     # optional: inspect full JSON-like results
#     # import json
#     # print(json.dumps(results, indent=2))
#
# asyncio.run(main())


import os
os.environ["OMP_NUM_THREADS"] = "8"       # match your physical core count
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"


from paddlex.inference import load_pipeline_config

cfg = load_pipeline_config("OCR")
cfg["use_doc_preprocessor"] = False
cfg["use_textline_orientation"] = False

for sub_name in ("TextDetection", "TextRecognition"):
    sub_cfg = cfg["SubModules"][sub_name]
    sub_cfg["use_hpip"] = True
    sub_cfg["hpi_config"] = {
        "auto_config": False,
        "backend": "openvino",
        "backend_config": {
            "cpu_num_threads": 8,   # adjust to your CPU core count
            "precision": "FP16",
        },
        "auto_paddle2onnx": True,
    }






