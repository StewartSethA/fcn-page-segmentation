DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running from DIR" $DIR

dataset="Barrett/A"
root_data_dir="../../data"

# python experiments/training_set_size_variation/create_subimage_datasets.py $root_data_dir/$dataset

# TODO: Refactor train.py to include experimental output folder as third argument!!!

for subdiv in 2 3 4 5 8 16; do
    pushd $DIR
    experiment_output_folder=$dataset/subdiv_${subdiv}
    mkdir -p $experiment_output_folder
    valdir="data/$dataset/subdivs_${subdiv}"
    traindir="${valdir}_train"
    pushd ../..
    python train.py ${traindir} ${valdir} | tee $DIR/${experiment_output_folder}/training.log
    popd
    mv ../../*.png $experiment_output_folder
    mv *.json $experiment_output_folder
    mv *.h5 $experiment_output_folder
    mv ${traindir}/*pred* $experiment_output_folder
    mv *.log $experiment_output_folder
    popd
done
