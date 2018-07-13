DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running from DIR" $DIR

dataset="Barrett/A"
notdataset="Barrett/notA"
root_data_dir="../../data"

# python experiments/training_set_size_variation/create_subimage_datasets.py $root_data_dir/$dataset

# TODO: Refactor train.py to include experimental output folder as third argument!!!
for subdiv in 8; do #2 3 4 5 8 16; do
    pushd $DIR
    #experiment_output_folder="experiments/training_set_size_variation/$dataset/subdiv_${subdiv}_lr_${lr}/validation"
    experiment_folder="experiments/training_set_size_variation/$dataset/subdiv_${subdiv}"
    experiment_output_folder="experiments/training_set_size_variation/$dataset/subdiv_${subdiv}/validation"
    mkdir -p $experiment_output_folder
    valdir="data/$notdataset"
    testdir=$valdir
    traindir="data/$dataset"
    
    pushd ../..

    echo ""
    echo ""    
    echo ""
    echo ""
    echo "Validating model", $experiment_folder/model_checkpoint.h5, "on data in", $testdir, " and outputting to", ${experiment_output_folder}
    python validate.py --test_folder $testdir --training_folder ${traindir} --validation_folder ${valdir} --log_dir ${experiment_output_folder} --output_folder ${experiment_output_folder} --framework tensorflow --model_type build_simple_hourglass --load_model_path "experiments/training_set_size_variation/Barrett_0/A/subdiv_${subdiv}/model_checkpoint999" | tee $DIR/${experiment_output_folder}/validation.log
    popd
    popd
done
