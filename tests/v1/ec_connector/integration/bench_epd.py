# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark EPD: TTFT under different encoder input sizes, and GPU memory.

Prerequisite: EPD service is running (start_epd_service.sh).
For each image:
  - cold request: clear EC cache, request once (encoder computes, producer saves)
  - warm request: same image again (cache hit, consumer loads from disk)
Measures TTFT, total latency, GPU memory snapshot, cache files on disk.
"""
import json
import os
import subprocess
import time

import openai
import requests

PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:10001")
MODEL = os.environ.get("MODEL", "/home/i26440/Qwen/Qwen3-VL-4B-Instruct")
EC_STORAGE = os.environ.get("EC_SHARED_STORAGE_PATH", "/tmp/ec_cache_test")

IMG_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES = [
    ("small_128.jpg", "~100"),
    ("mid_512.jpg", "~576"),
    ("large_2048.jpg", "~2025"),
]
MAX_TOKENS = 64


def clear_ec_cache() -> None:
    """清掉所有 EC cache,让下次请求必走冷启动."""
    os.system(f"rm -rf {EC_STORAGE}/*")


def list_cache_files() -> list:
    """列出已写盘的 encoder_cache.safetensors."""
    if not os.path.isdir(EC_STORAGE):
        return []
    found = []
    for d in os.listdir(EC_STORAGE):
        fp = os.path.join(EC_STORAGE, d, "encoder_cache.safetensors")
        if os.path.exists(fp):
            found.append(d)
    return found


def gpu_mem_snapshot() -> str:
    """抓 mx-smi 输出."""
    try:
        return subprocess.check_output(["mx-smi"], text=True, timeout=10)
    except Exception as e:  # noqa: BLE001
        return f"mx-smi failed: {e}"


def measure_ttft(image_path: str, max_tokens: int = MAX_TOKENS):
    """发 streaming 请求,测 TTFT 和总耗时."""
    # base_url 必须带 /v1 后缀,与 test_epd_correctness.py 保持一致
    client = openai.OpenAI(api_key="EMPTY", base_url=f"{PROXY_URL}/v1")
    msg = {
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"file://{image_path}"}},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[msg],
        max_tokens=max_tokens,
        stream=True,
    )
    first_token_t = None
    n_tokens = 0
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_t is None:
                first_token_t = time.perf_counter() - t0
            n_tokens += 1
    total_t = time.perf_counter() - t0
    return first_token_t, total_t, n_tokens


def check_proxy() -> bool:
    try:
        r = requests.get(f"{PROXY_URL}/v1/models", timeout=5)
        if r.status_code != 200:
            print(f"ERROR: proxy returned {r.status_code}")
            return False
        print("Proxy is ready.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot reach proxy at {PROXY_URL}: {e}")
        print("Start service first:")
        print("  bash tests/v1/ec_connector/integration/start_epd_service.sh")
        return False


def main() -> None:
    print(f"=== EPD Benchmark ===")
    print(f"Proxy:    {PROXY_URL}")
    print(f"Model:    {MODEL}")
    print(f"EC cache: {EC_STORAGE}")
    print(f"Max out tokens per request: {MAX_TOKENS}")

    if not check_proxy():
        return

    print("\n=== Generating test images ===")
    subprocess.check_call(
        ["python", os.path.join(IMG_DIR, "gen_test_images.py")]
    )

    results = []
    for img_name, expected_tok in IMAGES:
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"Skip {img_name}: not found")
            continue

        print(f"\n=== {img_name} ({expected_tok} encoder tokens) ===")

        # Cold: clear cache first
        clear_ec_cache()
        time.sleep(1)
        print("--- Before cold request (GPU memory) ---")
        print(gpu_mem_snapshot())

        ttft_cold, total_cold, n_cold = measure_ttft(img_path)
        print(
            f"Cold:  TTFT={ttft_cold * 1000:.0f}ms, "
            f"total={total_cold * 1000:.0f}ms, tokens={n_cold}"
        )
        time.sleep(1)
        print("--- After cold request (GPU memory) ---")
        print(gpu_mem_snapshot())

        cache_files = list_cache_files()
        print(f"Cache files on disk: {len(cache_files)} -> {cache_files}")

        # Warm: same image again, cache hit
        ttft_warm, total_warm, n_warm = measure_ttft(img_path)
        print(
            f"Warm:  TTFT={ttft_warm * 1000:.0f}ms, "
            f"total={total_warm * 1000:.0f}ms, tokens={n_warm}"
        )
        time.sleep(1)
        print("--- After warm request (GPU memory) ---")
        print(gpu_mem_snapshot())

        speedup = ttft_cold / ttft_warm if ttft_warm > 0 else 0
        print(f"TTFT speedup: {speedup:.2f}x")

        results.append({
            "image": img_name,
            "expected_tokens": expected_tok,
            "ttft_cold_ms": ttft_cold * 1000,
            "ttft_warm_ms": ttft_warm * 1000,
            "total_cold_ms": total_cold * 1000,
            "total_warm_ms": total_warm * 1000,
            "speedup": speedup,
            "cache_files": len(cache_files),
        })

    print("\n=== Summary ===")
    header = (
        f"{'Image':<20} {'Tokens':<10} "
        f"{'ColdTTFT(ms)':<15} {'WarmTTFT(ms)':<15} "
        f"{'Speedup':<10} {'Cache':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['image']:<20} {r['expected_tokens']:<10} "
            f"{r['ttft_cold_ms']:<15.0f} {r['ttft_warm_ms']:<15.0f} "
            f"{r['speedup']:<10.2f} {r['cache_files']:<8}"
        )

    out_json = "/tmp/epd_bench_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
