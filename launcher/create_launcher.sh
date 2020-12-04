#! /bin/bash
export PYTHONPATH=$PYTHONPATH:/home/frank/Duckietown/NewObstacle/NerualNetworkFun/dt-cnn
python3 util.py --image_dir duckietownDB/frames/ --annotation_dir duckietownDB/annotation/final_anns.json 
