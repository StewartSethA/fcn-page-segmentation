DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running from DIR" $DIR

root_data_dir="../../data"

# TODO: Refactor train.py to include experimental output folder as third argument!!!
lr=0.0005
for author in Barrett Iain Zhihan; do
    for leaveout in A B C; do
        # SUBSIZE mode only:
        #python experiments/training_set_size_variation/create_subimage_datasets.py $root_data_dir/$dataset
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
        experiment_output_folder="experiments/training_set_size_variation/$author/not$leaveout"
        mkdir -p $experiment_output_folder
        valdir="data/$author/$leaveout"
        traindir="data/$author/not$leaveout"
        pushd ../..
        python train.py --training_folder ${traindir} --validation_folder ${valdir} --log_dir ${experiment_output_folder} --lr $lr | tee $DIR/${experiment_output_folder}/training.log
        popd
        popd
    done
done

