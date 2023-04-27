import logging
import os
import re
from datetime import datetime
from training.consts import DEFAULT_INPUT_MODEL, SUGGESTED_INPUT_MODELS
from training.trainer import load_training_dataset, load_tokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# Cache data and tokenizer locally before creating a bunch of deepspeed processes and make sure they succeeds.
load_training_dataset()
load_tokenizer()

timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
input_model = "bigscience/bloomz-7b1-mt"
model_name = f"bloomz7b_dolly"
checkpoint_dir_name = f"{model_name}__{timestamp}"

root_path = os.getcwd()
deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config.json")

dolly_training_dir_name = "dolly_training"

# Use the local training root path if it was provided.  Otherwise try to find a sensible default.
local_training_root = "/mnt/task_runtime/trained_models"
dbfs_output_root = f"{local_training_root}/{dolly_training_dir_name}"

os.makedirs(local_training_root, exist_ok=True)
os.makedirs(dbfs_output_root, exist_ok=True)

local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)
dbfs_output_dir = os.path.join(dbfs_output_root, checkpoint_dir_name)

num_gpus = 2
num_gpus_flag = f"--num_gpus={num_gpus}"

tensorboard_display_dir = f"{local_output_dir}/runs"

print(f"Local Output Dir: {local_output_dir}")
print(f"DBFS Output Dir: {dbfs_output_dir}")
print(f"Tensorboard Display Dir: {tensorboard_display_dir}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
