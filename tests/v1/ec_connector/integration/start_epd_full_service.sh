#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 只启动 1E + 1P + 1D + proxy,不跑 correctness 测试,服务保持运行.
# 用法:
#   GPU_E=0 GPU_P=1 GPU_D=2 GPU_MEMORY_UTILIZATION=0.2 \
#     bash tests/v1/ec_connector/integration/start_epd_full_service.sh
# 停止: Ctrl+C 或 kill 进程

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

MODEL="${MODEL:-/home/i26440/Qwen/Qwen3-VL-4B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-2}"
DEVICE_AFFINITY_ENV="${DEVICE_AFFINITY_ENV:-CUDA_VISIBLE_DEVICES}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10002}"
EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache_test}"
LOG_PATH="${LOG_PATH:-/tmp}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

PIDS=()
mkdir -p "$LOG_PATH"
trap 'echo "[stop] killing 1E1P1D services..."; kill $(jobs -pr) 2>/dev/null' SIGINT SIGTERM EXIT

wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:${port}/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

echo "================================"
echo "Starting EPD 1E + 1P + 1D service"
echo "  Model:   $MODEL"
echo "  Encoder: GPU=$GPU_E, port=$ENCODE_PORT"
echo "  Prefill: GPU=$GPU_P, port=$PREFILL_PORT  (KV producer, EC consumer)"
echo "  Decode:  GPU=$GPU_D, port=$DECODE_PORT   (KV consumer)"
echo "  Proxy:   port=$PROXY_PORT"
echo "  EC cache: $EC_SHARED_STORAGE_PATH"
echo "  GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "================================"

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

# Encoder instance (EC producer)
env "$DEVICE_AFFINITY_ENV=$GPU_E" vllm serve "$MODEL" \
    --port "$ENCODE_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --gpu-memory-utilization 0.01 \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    > "$LOG_PATH"/1e1p1d_encoder.log 2>&1 &
PIDS+=($!)

# Prefill instance (EC consumer + KV producer via Nixl)
env "$DEVICE_AFFINITY_ENV=$GPU_P" \
VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
vllm serve "$MODEL" \
    --port "$PREFILL_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-request-id-headers \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer"
    }' \
    > "$LOG_PATH"/1e1p1d_prefill.log 2>&1 &
PIDS+=($!)

# Decode instance (KV consumer via Nixl)
env "$DEVICE_AFFINITY_ENV=$GPU_D" \
VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
vllm serve "$MODEL" \
    --port "$DECODE_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-request-id-headers \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}/tests/v1/ec_connector/integration" \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }' \
    > "$LOG_PATH"/1e1p1d_decode.log 2>&1 &
PIDS+=($!)

echo "Waiting for encoder..."
wait_for_server "$ENCODE_PORT"
echo "Waiting for prefill..."
wait_for_server "$PREFILL_PORT"
echo "Waiting for decode..."
wait_for_server "$DECODE_PORT"

# EPD proxy (encode -> prefill -> decode)
python "${GIT_ROOT}/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    > "$LOG_PATH"/1e1p1d_proxy.log 2>&1 &
PIDS+=($!)

echo "Waiting for proxy..."
wait_for_server "$PROXY_PORT"

curl -s http://127.0.0.1:"$PROXY_PORT"/v1/models
echo ""

echo "================================"
echo "1E+1P+1D service up. Now run benchmark:"
echo "  PROXY_URL=http://localhost:$PROXY_PORT \\"
echo "  python tests/v1/ec_connector/integration/bench_epd.py"
echo "Press Ctrl+C here to stop services."
echo "================================"

wait
