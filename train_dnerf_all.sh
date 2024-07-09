# train all data of D-NeRF dataset
# LIST=("hellwarrior" "mutant" "hook" "bouncingballs" "lego" "trex" "standup" "jumpingjacks")
LIST=("hellwarrior" "mutant" "bouncingballs" "trex" "standup" "jumpingjacks")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  CUDA_LAUNCH_BLOCKING=1 python train.py -s ../../data/dnerf/${ELEMENT} -m output/exp-demo-${ELEMENT} --eval --is_blender
  done