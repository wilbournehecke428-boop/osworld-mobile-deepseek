#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

export HF_HOME="$PWD/.hf-cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120

hf_cli="${GUI_OWL_HF_CLI:-../UI-S1-mac/.venv312/bin/hf}"
revision=3f061c2c562cc860c42bf32542a70e07a7ff4840
model=mPLUG/GUI-Owl-1.5-4B-Instruct

downloaded_bytes() {
    find weights -type f \( -name "$1" -o -name "*.$2.incomplete" \) \
        -exec stat -f '%z' {} + | awk '{ total += $1 } END { printf "%.0f\n", total }'
}

first=model-00001-of-00002.safetensors
second=model-00002-of-00002.safetensors
first_sha=03c36699f9437d781a6d4d418afe69ce2e198947c234819b41a10e8c29cf7ea4
second_sha=31e6c27c212658b5faea4a72442047a5ce8d4fbd409b775fa18752fb5a20ea09

download_shard() {
    local shard="$1" digest="$2" current_bytes previous_bytes stalls=0
    previous_bytes="$(downloaded_bytes "$shard" "$digest")"
    if [[ ! -f "weights/$shard" ]]; then
        until "$hf_cli" download "$model" "$shard" --revision "$revision" --local-dir weights; do
            current_bytes="$(downloaded_bytes "$shard" "$digest")"
            if (( current_bytes > previous_bytes )); then
                stalls=0
            else
                stalls=$((stalls + 1))
            fi
            if (( stalls >= 6 )); then
                echo "$shard stopped advancing after six consecutive attempts." >&2
                return 1
            fi
            previous_bytes="$current_bytes"
            echo "Retrying $shard from its partial file in 15 seconds." >&2
            sleep 15
        done
    fi
    (cd weights && shasum -a 256 -c <<< "$digest  $shard")
}

download_shard "$first" "$first_sha" &
first_pid=$!
download_shard "$second" "$second_sha" &
second_pid=$!
status=0
wait "$first_pid" || status=1
wait "$second_pid" || status=1
exit "$status"
