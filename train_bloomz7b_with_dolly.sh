NUM_GPUS=2
NUM_GPUS_FLAG="--num_gpus=${NUM_GPUS}"

# train parameters
INPUT_MODEL="bigscience/bloomz-7b1-mt"
MODEL_NAME="bloomz7b_dolly"
DEEPSPEED_CONFIG="config/ds_bloom7_fp16_config.json"
MAX_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
WARMUP_STEPS=50
LREARNING_RATE=5e-6

LOGGING_STEPS=10
SAVE_STEP=200
SAVE_TOTAL_LIMIT=20
EVAL_STEPS=50
TEST_STEPS=200

# output locations
TRAIN_ROOT_DIR="/mnt/task_runtime/trained_models"
TIMESTAMP=$(date +"%Y-%m-%dT%H:%M:%S")
CHECKPOINT_DIR_NAME="${MODEL_NAME}__${TIMESTAMP}"
LOCAL_OUTPUT_DIR="${TRAIN_ROOT_DIR}/${CHECKPOINT_DIR_NAME}"
DBFS_OUTPUT_DIR="${TRAIN_ROOT_DIR}/dolly_training/${CHECKPOINT_DIR_NAME}"

# run with deepspeed
deepspeed ${NUM_GPUS_FLAG} \
  --module training.trainer \
  --input-model ${INPUT_MODEL} \
  --deepspeed ${DEEPSPEED_CONFIG} \
  --epochs ${MAX_EPOCHS} \
  --local-output-dir ${LOCAL_OUTPUT_DIR} \
  --dbfs-output-dir ${DBFS_OUTPUT_DIR} \
  --per-device-train-batch-size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --per-device-eval-batch-size ${PER_DEVICE_EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --logging-steps ${LOGGING_STEPS} \
  --save-steps ${SAVE_STEP} \
  --save-total-limit ${SAVE_TOTAL_LIMIT} \
  --eval-steps ${EVAL_STEPS} \
  --warmup-steps ${WARMUP_STEPS} \
  --test-size ${TEST_STEPS} \
  --lr ${LREARNING_RATE}