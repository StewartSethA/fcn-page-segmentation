base=$1

for dataset in Barrett Iain Zhihan; do
    base_folder=/home/ubuntu/workspace/fcn-page-segmentation/data/
    for train_instance in A B C; do
          train_folder="${base_folder}/${dataset}/$train_instance"

          echo "Training ${train_folder}..."
          experiment_name="${dataset}${train_instance}"
          expfolder="experiments/${dataset}/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${dataset}${train_instance}
          echo "Using training model ${model_path}..."
          lr=0.005
          python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr | tee ${expfolder}/training.log
          #python transcriber_trainer_workflow.py "${model_path}" "${train_folder}" train |  tee ${expfolder}/training_log.txt || exit

    done
done
