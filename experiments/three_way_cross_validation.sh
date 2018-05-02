base=$1

for dataset in Barrett Iain Zhihan; do
    base_folder=/home/ubuntu/workspace/document-layout-analysis/data/Forms/
    for train_instance in A B C; do
      train_folder="${base_folder}/${dataset}/$train_instance"

        for validation_dataset in Barrett Iain Zhihan; do
            for validation_instance in A B C; do
              echo "Using training model ${train_folder}..."
              experiment_name="${dataset}${train_instance}"
              expfolder="experiments/${dataset}/${experiment_name}/"
              validation_name="${validation_dataset}${validation_instance}"
              validation_results_folder="${expfolder}${validation_name}"
              mkdir -p $validation_results_folder
              validation_folder="${base_folder}/${validation_dataset}/$validation_instance"
              python transcriber_trainer_workflow.py "${train_folder}" "${validation_folder}" # |  tee validation_log.txt || exit
              
              echo "Moving validation results to folder:" "${validation_results_folder}"
              mv ${validation_folder}/*.jpg_* "${validation_results_folder}"/
              mv validation_log.txt "${validation_results_folder}"/
              
            done
        done
    done
done
