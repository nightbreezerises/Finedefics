#!/bin/bash
# ============================================================================
# Finedefics 评估脚本
# 功能: 在指定数据集上评估 Finedefics 模型的分类准确率
# 用法: ./run_evaluate.sh [config_file]
# 
# 支持的数据集格式: dog, bird, flower, car, pet, aircraft, food, birdsnap,
#   caltech101, caltech256, dtd, eurosat, ucf, sun397, deepfashion_multimodal,
#   imagenet_1k, imagenet_a, imagenet_r, imagenet_sketch, imagenet_v2
# ============================================================================

### 用法
# nohup /home/hdl/project/Finedefics/scripts/run_evaluate.sh > /home/hdl/project/Finedefics/logs/run.log 2>&1 &

set -e  # 遇到错误立即退出

# ============================================================================
# 1. 路径设置 (使用相对路径，工作目录为 scripts 所在的项目根目录)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 配置文件路径
CONFIG_FILE="${1:-${SCRIPT_DIR}/config.yaml}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

echo "=============================================="
echo "Finedefics 评估脚本"
echo "=============================================="
echo "[INFO] 项目根目录: ${PROJECT_ROOT}"
echo "[INFO] 配置文件: ${CONFIG_FILE}"

# ============================================================================
# 2. 解析 YAML 配置文件
# ============================================================================
# 简单的 YAML 解析函数 (不依赖外部工具)
parse_yaml() {
    local yaml_file=$1
    local key=$2
    local result=""
    
    # 处理嵌套键，如 paths.data_dir
    if [[ "$key" == *"."* ]]; then
        local parent="${key%%.*}"
        local child="${key#*.}"
        local in_parent=false
        
        while IFS= read -r line || [[ -n "$line" ]]; do
            # 跳过注释和空行
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// }" ]] && continue
            
            # 检查是否进入父级
            if [[ "$line" =~ ^${parent}: ]]; then
                in_parent=true
                continue
            fi
            
            # 如果在父级内，检查子键
            if $in_parent; then
                # 如果遇到新的顶级键，退出
                if [[ "$line" =~ ^[a-zA-Z_] ]] && [[ ! "$line" =~ ^[[:space:]] ]]; then
                    break
                fi
                # 匹配子键
                if [[ "$line" =~ ^[[:space:]]+${child}:[[:space:]]*(.*) ]]; then
                    result="${BASH_REMATCH[1]}"
                    # 去除引号和注释
                    result="${result%%#*}"
                    result="${result%\"}"
                    result="${result#\"}"
                    result="${result%\'}"
                    result="${result#\'}"
                    # 去除首尾空格
                    result="${result#"${result%%[![:space:]]*}"}"
                    result="${result%"${result##*[![:space:]]}"}"
                    break
                fi
            fi
        done < "$yaml_file"
    else
        # 顶级键
        while IFS= read -r line || [[ -n "$line" ]]; do
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            if [[ "$line" =~ ^${key}:[[:space:]]*(.*) ]]; then
                result="${BASH_REMATCH[1]}"
                result="${result%%#*}"
                result="${result%\"}"
                result="${result#\"}"
                result="${result%\'}"
                result="${result#\'}"
                result="${result#"${result%%[![:space:]]*}"}"
                result="${result%"${result##*[![:space:]]}"}"
                break
            fi
        done < "$yaml_file"
    fi
    
    echo "$result"
}

# 解析带引号的值（用于提示词）
parse_yaml_quoted() {
    local yaml_file=$1
    local parent=$2
    local child=$3
    local result=""
    local in_parent=false
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        
        if [[ "$line" =~ ^${parent}: ]]; then
            in_parent=true
            continue
        fi
        
        if $in_parent; then
            if [[ "$line" =~ ^[a-zA-Z_] ]] && [[ ! "$line" =~ ^[[:space:]] ]]; then
                break
            fi
            if [[ "$line" =~ ^[[:space:]]+${child}:[[:space:]]*\"(.*)\" ]]; then
                result="${BASH_REMATCH[1]}"
                break
            elif [[ "$line" =~ ^[[:space:]]+${child}:[[:space:]]*\'(.*)\' ]]; then
                result="${BASH_REMATCH[1]}"
                break
            fi
        fi
    done < "$yaml_file"
    
    echo "$result"
}

# ============================================================================
# 3. 读取配置参数
# ============================================================================
echo "[INFO] 解析配置文件..."

# 环境配置
CONDA_PATH=$(parse_yaml "${CONFIG_FILE}" "environment.conda_path")
CONDA_ENV=$(parse_yaml "${CONFIG_FILE}" "environment.conda_env")

# 路径配置
DATA_DIR=$(parse_yaml "${CONFIG_FILE}" "paths.data_dir")
LOG_DIR=$(parse_yaml "${CONFIG_FILE}" "paths.log_dir")
RESULTS_DIR=$(parse_yaml "${CONFIG_FILE}" "paths.results_dir")
FINEDEFICS_DIR=$(parse_yaml "${CONFIG_FILE}" "paths.finedefics_dir")

# GPU 配置
GPU=$(parse_yaml "${CONFIG_FILE}" "gpu")

# 数据集
DATASET=$(parse_yaml "${CONFIG_FILE}" "dataset")

# 分类评估参数
TEST_RATIO=$(parse_yaml "${CONFIG_FILE}" "classification.test_ratio")
BATCH_SIZE=$(parse_yaml "${CONFIG_FILE}" "classification.batch_size")
MAX_EXAMPLES=$(parse_yaml "${CONFIG_FILE}" "classification.max_examples")
TASK=$(parse_yaml "${CONFIG_FILE}" "classification.task")
CHOICE_ENUM=$(parse_yaml "${CONFIG_FILE}" "classification.choice_enumeration")
LOAD_QUANTIZED=$(parse_yaml "${CONFIG_FILE}" "classification.load_quantized")

# 实验配置
EXPERIMENT_ID=$(parse_yaml "${CONFIG_FILE}" "experiment.id")
EXPERIMENT_NAME=$(parse_yaml "${CONFIG_FILE}" "experiment.name")

# ============================================================================
# 4. 设置默认值和数据集映射
# ============================================================================
CONDA_PATH="${CONDA_PATH:-/home/hdl/miniconda3}"
CONDA_ENV="${CONDA_ENV:-finedefics}"
DATA_DIR="${DATA_DIR:-./datasets}"
LOG_DIR="${LOG_DIR:-./logs}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
FINEDEFICS_DIR="${FINEDEFICS_DIR:-./models/Finedefics}"
GPU="${GPU:-0}"
DATASET="${DATASET:-dog}"
TEST_RATIO="${TEST_RATIO:-100.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-9999}"
TASK="${TASK:-mc}"
CHOICE_ENUM="${CHOICE_ENUM:-ABCD}"
LOAD_QUANTIZED="${LOAD_QUANTIZED:-false}"
EXPERIMENT_ID="${EXPERIMENT_ID:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

# 支持的数据集列表
SUPPORTED_DATASETS="dog bird flower car pet aircraft food birdsnap caltech101 caltech256 dtd eurosat ucf sun397 imagenet_1k imagenet_a imagenet_r imagenet_sketch imagenet_v2"

# 数据集目录映射 (config数据集名 -> 本地目录名)
declare -A DATASET_DIR_MAP
DATASET_DIR_MAP["dog"]="dogs_120"
DATASET_DIR_MAP["bird"]="CUB_200_2011"
DATASET_DIR_MAP["flower"]="flowers_102"
DATASET_DIR_MAP["car"]="car_196"
DATASET_DIR_MAP["pet"]="pet_37"
DATASET_DIR_MAP["aircraft"]="fgvc_aircraft"
DATASET_DIR_MAP["food"]="food_101"
DATASET_DIR_MAP["birdsnap"]="birdsnap"
DATASET_DIR_MAP["caltech101"]="caltech101"
DATASET_DIR_MAP["caltech256"]="caltech256"
DATASET_DIR_MAP["dtd"]="dtd"
DATASET_DIR_MAP["eurosat"]="eurosat"
DATASET_DIR_MAP["ucf"]="ucf101"
DATASET_DIR_MAP["sun397"]="SUN397"
DATASET_DIR_MAP["imagenet_1k"]="ImageNet_1k"
DATASET_DIR_MAP["imagenet_a"]="ImageNet_A"
DATASET_DIR_MAP["imagenet_r"]="ImageNet_R"
DATASET_DIR_MAP["imagenet_sketch"]="ImageNet_Sketch"
DATASET_DIR_MAP["imagenet_v2"]="ImageNet_v2"

# FOCI-Benchmark 数据集名映射 (用于调用 run_ic_bench.py)
declare -A FOCI_DATASET_MAP
FOCI_DATASET_MAP["dog"]="dog-120"
FOCI_DATASET_MAP["bird"]="bird-200"
FOCI_DATASET_MAP["flower"]="flower-102"
FOCI_DATASET_MAP["car"]="car-196"
FOCI_DATASET_MAP["pet"]="pet-37"
FOCI_DATASET_MAP["aircraft"]="aircraft-102"
FOCI_DATASET_MAP["food"]="food101"

# 提示词映射
declare -A PROMPT_MAP
PROMPT_MAP["dog"]="Which of these dogs is shown in the image?"
PROMPT_MAP["bird"]="Which of these birds is shown in the image?"
PROMPT_MAP["flower"]="Which of these flowers is shown in the image?"
PROMPT_MAP["car"]="Which of these cars is shown in the image?"
PROMPT_MAP["pet"]="Which of these pets is shown in the image?"
PROMPT_MAP["aircraft"]="Which of these aircrafts is shown in the image?"
PROMPT_MAP["food"]="Which of these foods is shown in the image?"
PROMPT_MAP["birdsnap"]="Which of these birds is shown in the image?"
PROMPT_MAP["caltech101"]="Which of these objects is shown in the image?"
PROMPT_MAP["caltech256"]="Which of these objects is shown in the image?"
PROMPT_MAP["dtd"]="Which of these textures is shown in the image?"
PROMPT_MAP["eurosat"]="Which of these satellite images is shown in the image?"
PROMPT_MAP["ucf"]="Which of these actions is shown in the image?"
PROMPT_MAP["sun397"]="Which of these scenes is shown in the image?"
PROMPT_MAP["imagenet_1k"]="Which of these objects is shown in the image?"
PROMPT_MAP["imagenet_a"]="Which of these objects is shown in the image?"
PROMPT_MAP["imagenet_r"]="Which of these objects is shown in the image?"
PROMPT_MAP["imagenet_sketch"]="Which of these objects is shown in the image?"
PROMPT_MAP["imagenet_v2"]="Which of these objects is shown in the image?"

# 获取数据集对应的本地目录名
DATASET_LOCAL_DIR="${DATASET_DIR_MAP[${DATASET}]}"

# 获取 FOCI-Benchmark 数据集名（如果有映射）
FOCI_DATASET_NAME="${FOCI_DATASET_MAP[${DATASET}]:-}"

# 获取数据集对应的提示词
PROMPT_QUERY="${PROMPT_MAP[${DATASET}]}"

# ============================================================================
# 5. 验证配置
# ============================================================================
echo ""
echo "[INFO] 配置参数:"
echo "  - Conda 路径: ${CONDA_PATH}"
echo "  - Conda 环境: ${CONDA_ENV}"
echo "  - 数据集目录: ${DATA_DIR}"
echo "  - 日志目录: ${LOG_DIR}"
echo "  - 结果目录: ${RESULTS_DIR}"
echo "  - 模型目录: ${FINEDEFICS_DIR}"
echo "  - GPU: ${GPU}"
echo "  - 数据集: ${DATASET}"
echo "  - 本地数据集目录: ${DATASET_LOCAL_DIR}"
echo "  - FOCI数据集名: ${FOCI_DATASET_NAME:-N/A}"
echo "  - 测试比例: ${TEST_RATIO}%"
echo "  - 批次大小: ${BATCH_SIZE}"
echo "  - 最大样本数: ${MAX_EXAMPLES}"
echo "  - 任务类型: ${TASK}"
echo "  - 量化加载: ${LOAD_QUANTIZED}"
echo "  - 实验编号: ${EXPERIMENT_ID}"
echo "  - 实验名称: ${EXPERIMENT_NAME:-无}"
echo "  - 提示词: ${PROMPT_QUERY}"
echo ""

# 验证数据集是否支持
if [[ ! " ${SUPPORTED_DATASETS} " =~ " ${DATASET} " ]]; then
    echo "[ERROR] 不支持的数据集: ${DATASET}"
    echo "[INFO] 支持的数据集: ${SUPPORTED_DATASETS}"
    exit 1
fi

# 验证模型目录
MODEL_PATH="${PROJECT_ROOT}/${FINEDEFICS_DIR}"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] 模型目录不存在: ${MODEL_PATH}"
    exit 1
fi

# 验证数据集目录
IMAGE_ROOT="${PROJECT_ROOT}/${DATA_DIR}/${DATASET_LOCAL_DIR}"
if [[ ! -d "${IMAGE_ROOT}" ]]; then
    echo "[ERROR] 数据集目录不存在: ${IMAGE_ROOT}"
    exit 1
fi

# 验证 split_datasets_images.json 文件
SPLIT_FILE="${IMAGE_ROOT}/images_split/split_datasets_images.json"
if [[ ! -f "${SPLIT_FILE}" ]]; then
    echo "[ERROR] 数据集划分文件不存在: ${SPLIT_FILE}"
    exit 1
fi
echo "[INFO] 数据集划分文件: ${SPLIT_FILE}"

# ============================================================================
# 6. 创建日志目录和结果目录
# ============================================================================
# 日志目录: ./logs/<dataset>/run_evaluate.log (按要求的格式)
LOG_SUBDIR="${PROJECT_ROOT}/${LOG_DIR}/${DATASET}"
mkdir -p "${LOG_SUBDIR}"

# 结果目录: ./results/exp<id>/
RESULTS_SUBDIR="${PROJECT_ROOT}/${RESULTS_DIR}/exp${EXPERIMENT_ID}"
mkdir -p "${RESULTS_SUBDIR}"

# 日志文件名: ./logs/<dataset>/run_evaluate.log
LOG_FILE="${LOG_SUBDIR}/run_evaluate.log"

echo "[INFO] 日志文件: ${LOG_FILE}"
echo "[INFO] 结果目录: ${RESULTS_SUBDIR}"

# ============================================================================
# 7. 日志函数 (实时写入)
# ============================================================================
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [exp${EXPERIMENT_ID}] $1"
    echo "${msg}" | tee -a "${LOG_FILE}"
}

log_header() {
    echo "" | tee -a "${LOG_FILE}"
    echo "==============================================================" | tee -a "${LOG_FILE}"
    log "$1"
    echo "==============================================================" | tee -a "${LOG_FILE}"
}

# ============================================================================
# 8. 开始评估
# ============================================================================
log_header "Finedefics 评估开始"
log "实验编号: ${EXPERIMENT_ID}"
log "实验名称: ${EXPERIMENT_NAME:-无}"
log "数据集: ${DATASET} (目录: ${DATASET_LOCAL_DIR})"
log "模型: ${MODEL_PATH}"
log "GPU: ${GPU}"
log "测试比例: ${TEST_RATIO}%"
log "数据集划分文件: ${SPLIT_FILE}"

# 激活 Conda 环境
log "激活 Conda 环境: ${CONDA_ENV}"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# 设置 GPU
export CUDA_VISIBLE_DEVICES="${GPU}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ============================================================================
# 9. 判断评估方式
# ============================================================================
# 如果数据集在 FOCI-Benchmark 支持列表中，使用 FOCI-Benchmark 评估
# 否则使用自定义评估脚本

if [[ -n "${FOCI_DATASET_NAME}" ]]; then
    # 使用 FOCI-Benchmark 评估
    log "使用 FOCI-Benchmark 进行评估"
    
    BENCHMARK_DIR="${PROJECT_ROOT}/FOCI-Benchmark"
    if [[ ! -d "${BENCHMARK_DIR}" ]]; then
        echo "[ERROR] FOCI-Benchmark 目录不存在: ${BENCHMARK_DIR}"
        exit 1
    fi
    
    cd "${BENCHMARK_DIR}"
    log "工作目录: $(pwd)"
    
    # 构建评估命令
    EVAL_CMD="python run_ic_bench.py"
    EVAL_CMD+=" --model=${MODEL_PATH}"
    EVAL_CMD+=" --dataset=${FOCI_DATASET_NAME}"
    EVAL_CMD+=" --image_root=${IMAGE_ROOT}"
    EVAL_CMD+=" --batchsize=${BATCH_SIZE}"
    EVAL_CMD+=" --max_examples=${MAX_EXAMPLES}"
    EVAL_CMD+=" --task=${TASK}"
    EVAL_CMD+=" --choice_enumeration=${CHOICE_ENUM}"
    EVAL_CMD+=" --results_output_folder=${RESULTS_SUBDIR}"
    EVAL_CMD+=" --prompt_query='${PROMPT_QUERY}'"
    
    # 量化加载
    if [[ "${LOAD_QUANTIZED}" != "false" ]] && [[ -n "${LOAD_QUANTIZED}" ]]; then
        EVAL_CMD+=" --load_quantized=${LOAD_QUANTIZED}"
    fi
    
    log "执行命令: ${EVAL_CMD}"
    echo "" | tee -a "${LOG_FILE}"
    
    # 执行评估（实时输出到终端和日志）
    log_header "评估输出"
    eval "${EVAL_CMD}" 2>&1 | tee -a "${LOG_FILE}"
    EVAL_EXIT_CODE=${PIPESTATUS[0]}
    
else
    # 使用自定义评估（基于 split_datasets_images.json）
    log "使用自定义评估脚本 (基于 split_datasets_images.json)"
    log "数据集 ${DATASET} 不在 FOCI-Benchmark 支持列表中"
    log "将使用 split_datasets_images.json 中的测试集进行评估"
    
    # TODO: 实现自定义评估逻辑
    # 当前先输出提示信息
    log "[WARNING] 自定义评估脚本尚未实现，请使用 FOCI-Benchmark 支持的数据集"
    log "FOCI-Benchmark 支持的数据集: dog, bird, flower, car, pet, aircraft, food"
    EVAL_EXIT_CODE=0
fi

# ============================================================================
# 10. 评估完成
# ============================================================================
echo "" | tee -a "${LOG_FILE}"
if [[ ${EVAL_EXIT_CODE} -eq 0 ]]; then
    log_header "评估完成"
    log "状态: 成功"
    log "结果保存在: ${RESULTS_SUBDIR}/"
    log "日志保存在: ${LOG_FILE}"
else
    log_header "评估失败"
    log "状态: 失败 (退出码: ${EVAL_EXIT_CODE})"
    log "请检查日志: ${LOG_FILE}"
    exit ${EVAL_EXIT_CODE}
fi

# 返回项目根目录
cd "${PROJECT_ROOT}"

echo ""
echo "[INFO] 评估脚本执行完毕"