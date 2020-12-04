#! /bin/bash
#? Set env
export PYTHONPATH=$PYTHONPATH:/home/frank/Duckietown/NewObstacle/NerualNetworkFun/dt-cnn

#? Training configuration
TYPE=1_mobile_net_v2
#CHECKPOINT_DIR=trials/${TYPE}/model_dir #Path to directory holding a checkpoint.  If `checkpoint_dir` is provided, this binary operates in eval-only mode, writing resulting metrics to `model_dir`.
CHECKPOINT_EVERY_N=1000 #Integer defining how often we checkpoint.
#--eval_timeout: Number of seconds to wait for anevaluation checkpoint before exiting.
MODEL_DIR=trials/${TYPE}/model_dir #Path to output model directory where event and checkpoint files will be written.
NUM_TRAIN_STEPS=15000 #Number of train steps.
PIPELINE_CONFIG_PATH=trials/${TYPE}/config/pipeline.config #Path to pipeline config file.

python3 object_detection/model_main_tf2.py \
  --checkpoint_every_n=$CHECKPOINT_EVERY_N \
  --model_dir=$MODEL_DIR \
  --num_train_steps=$NUM_TRAIN_STEPS \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr