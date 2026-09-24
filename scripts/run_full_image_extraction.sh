#!/usr/bin/env bash
# 在服务器后台顺序处理三个数据集；重复运行时从已有 JSONL 继续。
set -Eeuo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image_python="${TBPS_IMAGE_PYTHON:-/home/lzf/ldx/envs/tbps-image/bin/python}"
datasets_dir="${TBPS_DATASETS_DIR:-/home/lzf/TBPS/Datasets}"
upar_source="${TBPS_UPAR_SOURCE:-/home/lzf/ldx/projects/upar_challenge}"
checkpoint="${TBPS_UPAR_CHECKPOINT:-/home/lzf/ldx/checkpoints/idea-TBPS-test1/UPAR/best_model.pth}"
output_dir="${TBPS_IMAGE_OUTPUT_DIR:-/home/lzf/ldx/outputs/idea-TBPS-test1/upar/full}"
batch_size="${TBPS_IMAGE_BATCH_SIZE:-2}"

mkdir -p -- "$output_dir"

run_dataset() {
    local dataset="$1" annotation="$2" image_root="$3"
    printf '[%s] 开始 %s 全量提取\n' "$(date '+%F %T')" "$dataset"
    "$image_python" -u "$project_dir/scripts/batch_extract_image.py" \
        --dataset "$dataset" \
        --annotation "$annotation" \
        --image-root "$image_root" \
        --upar-source "$upar_source" \
        --checkpoint "$checkpoint" \
        --batch-size "$batch_size" \
        --resume \
        --progress-every 100 \
        --output "$output_dir/$dataset.jsonl"
    printf '[%s] 完成 %s\n' "$(date '+%F %T')" "$dataset"
}

run_dataset cuhk "$datasets_dir/CUHK-PEDES/reid_raw.json" "$datasets_dir/CUHK-PEDES/imgs"
run_dataset icfg "$datasets_dir/ICFG-PEDES/ICFG-PEDES.json" "$datasets_dir/ICFG-PEDES/imgs"
run_dataset rstp "$datasets_dir/RSTPReid/data_captions.json" "$datasets_dir/RSTPReid/imgs"
printf '[%s] 三个数据集全部完成\n' "$(date '+%F %T')"
