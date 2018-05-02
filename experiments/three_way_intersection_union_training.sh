base=$1

for dataset in Intersection Union; do
    base_folder=/home/ubuntu/workspace/doc_semantic_segmentation/data_intersection_union/
    for train_instance in A B C; do
      train_folder="${base_folder}/${dataset}/$train_instance"

          echo "Training ${train_folder}..."
          experiment_name="${dataset}${train_instance}"
          expfolder="experiments/${dataset}/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${dataset}${train_instance}
          echo "Using training model ${model_path}..."

          python transcriber_trainer_workflow.py "${model_path}" "${train_folder}" train |  tee ${expfolder}/training_log.txt || exit

          mv checkpoint "${expfolder}"/
          mv *.meta "${expfolder}"/
          mv *.index "${expfolder}"/
          mv *.data* "${expfolder}"/
          mv *.jpg "${expfolder}"/
          mv *.png "${expfolder}"/
          mv *.txt "${expfolder}"/
    done
done
