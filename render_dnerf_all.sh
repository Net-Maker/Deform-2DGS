# train all data of D-NeRF dataset
LIST=("hellwarrior" "mutant" "hook" "bouncingballs" "lego" "trex" "standup" "jumpingjacks")
# LIST=("hellwarrior" "mutant" "bouncingballs" "trex" "standup" "jumpingjacks")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  python render_2d.py -m output/exp-name-${ELEMENT} --mode render
  done