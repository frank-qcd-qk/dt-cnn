#! /bin/bash
TYPE=1_mobile_net_v2
export CONFIG_FILE=trials/${TYPE}/config/pipeline.config
export CHECKPOINT_PATH=trials/${TYPE}/final_model/
export OUTPUT_DIR=trials/${TYPE}/tflite_graph

python3 object_detection/export_tflite_graph_tf2.py \
--pipeline_config_path $CONFIG_FILE \
--trained_checkpoint_dir $CHECKPOINT_PATH \
--output_directory $OUTPUT_DIR \

python3 launcher/write_tflite.py