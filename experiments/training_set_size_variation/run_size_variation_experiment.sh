DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running from DIR" $DIR

dataset="Barrett/A"
root_data_dir="../../data"

# python experiments/training_set_size_variation/create_subimage_datasets.py $root_data_dir/$dataset

# TODO: Refactor train.py to include experimental output folder as third argument!!!
lr=0.0005
for subdiv in 1; do # 2 3 4 5 8 16; do
    pushd $DIR
    experiment_output_folder="experiments/training_set_size_variation/$dataset/subdiv_${subdiv}_lr_${lr}"
    mkdir -p $experiment_output_folder
    valdir="data/$dataset/subdivs_${subdiv}"
    traindir="${valdir}_train"
    pushd ../..
    crop_size=224
    python train.py --training_folder ${traindir} --validation_folder ${valdir} --log_dir ${experiment_output_folder} --lr $lr --crop_size $crop_size --framework tensorflow --model_type build_simple_hourglass --num_classes 5 --min_height_scale 0.25 --max_height_scale 4.0 | tee $DIR/${experiment_output_folder}/training.log
    popd
    #mv ../../*.png $experiment_output_folder
    #mv *.json $experiment_output_folder
    #mv *.h5 $experiment_output_folder
    #mv ${traindir}/*pred* $experiment_output_folder
    #mv *.log $experiment_output_folder
    popd
done
