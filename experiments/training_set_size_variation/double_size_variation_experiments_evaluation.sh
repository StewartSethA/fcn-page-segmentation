DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running from DIR" $DIR

dataset="Barrett/A"
root_data_dir="../../data"

# python experiments/training_set_size_variation/create_subimage_datasets.py $root_data_dir/$dataset

# TODO: Refactor train.py to include experimental output folder as third argument!!!
lr=0.0005
for author in Barrett Iain Zhihan; do
    for leaveout in A B C; do
        pushd $DIR
        data_dir=$root_data_dir/$author/not$leaveout
        pwd
        echo $data_dir
        mkdir -p $data_dir
        for leavein in A B C; do
            if [ "$leaveout" != "$leavein" ]; then
                cp $root_data_dir/$author/$leavein/* $data_dir/
            fi
        done
        experiment_output_folder="experiments/training_set_size_variation/$dataset/not$leaveout"
        mkdir -p $experiment_output_folder
        valdir="data/$author/$leaveout"
        traindir="data/$author/not$leaveout"
        pushd ../..
        python validate.py --training_folder ${traindir} --validation_folder ${valdir} --test_folder ${valdir} --output_dir ${experiment_output_folder} --load_model_path $experiment_output_folder/
        popd
        popd
    done
done

