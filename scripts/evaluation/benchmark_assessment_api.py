"""Benchmark a running Assessment API v2 service without retaining assessed text."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx


def _payload(text: str, request_id: str, deadline_ms: int = 1000) -> dict:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "request_id": request_id,
        "profile": "jentic_gateway_fast",
        "deadline_ms": deadline_ms,
        "content_sha256": digest,
        "segments": [
            {
                "id": "segment-1",
                "text": text,
                "source": "text",
                "locator": "$.body",
                "sha256": digest,
            }
        ],
        "context": {
            "surface": "runtime_response",
            "upstream_status": 200,
            "content_type": "text/plain",
        },
    }


async def _post(client: httpx.AsyncClient, payload: dict) -> tuple[int, float, dict]:
    started = time.perf_counter()
    response = await client.post("/v2/assessments", json=payload)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    try:
        body = response.json()
    except ValueError:
        body = {}
    return response.status_code, elapsed_ms, body


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[94]


async def benchmark(args) -> dict:
    headers = {"Authorization": f"Bearer {args.token}"}
    timeout = httpx.Timeout(max(5.0, args.deadline_ms / 1000 + 2))
    report: dict = {"latency": {}, "overload": {}}
    async with httpx.AsyncClient(
        base_url=args.endpoint.rstrip("/"), headers=headers, timeout=timeout
    ) as client:
        sizes = {"1_kib": 1024, "64_kib": 64 * 1024}
        for bucket, size in sizes.items():
            prefix = "A routine weather observation for Dublin. "
            text = (prefix * ((size // len(prefix)) + 1))[:size]
            results = []
            for index in range(args.iterations):
                results.append(
                    await _post(
                        client,
                        _payload(text, f"benchmark-{bucket}-{time.time_ns()}-{index}"),
                    )
                )
            statuses = Counter(status for status, _, _ in results)
            latencies = [elapsed for status, elapsed, _ in results if status == 200]
            service_times = [
                float(body["processing_time_ms"])
                for status, _, body in results
                if status == 200 and "processing_time_ms" in body
            ]
            report["latency"][bucket] = {
                "requests": len(results),
                "statuses": dict(statuses),
                "wall_p95_ms": _p95(latencies) if latencies else None,
                "service_p95_ms": _p95(service_times) if service_times else None,
                "verdicts": dict(
                    Counter(body.get("verdict") for status, _, body in results if status == 200)
                ),
            }

        overload_text = ("A bounded response body for queue saturation. " * 1600)[: 64 * 1024]
        overload_results = await asyncio.gather(
            *(
                _post(
                    client,
                    _payload(
                        overload_text,
                        f"benchmark-overload-{time.time_ns()}-{index}",
                        args.deadline_ms,
                    ),
                )
                for index in range(args.concurrency)
            )
        )
        overload_statuses = Counter(status for status, _, _ in overload_results)
        report["overload"] = {
            "concurrency": args.concurrency,
            "statuses": dict(overload_statuses),
            "bounded_rejections": overload_statuses[429] + overload_statuses[504],
        }

    latency_values = [entry["wall_p95_ms"] for entry in report["latency"].values()]
    report["passed"] = (
        all(value is not None and value <= 200.0 for value in latency_values)
        and report["overload"]["bounded_rejections"] > 0
        and not (set(report["overload"]["statuses"]) - {200, 429, 504})
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--deadline-ms", type=int, default=1000)
    args = parser.parse_args()
    report = asyncio.run(benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
