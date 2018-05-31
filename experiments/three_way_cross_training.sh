base=$1


# NEW! Trying out different models...

for dataset in Barrett Iain Zhihan; do
    base_folder=/home/ubuntu/workspace/fcn-page-segmentation/data/
    for train_instance in A B C; do
        for model_type in unet densenet_tiramisu template_matcher_single_hidden_layer build_simple_hourglass build_model_functional; do
          train_folder="${base_folder}/${dataset}/$train_instance"

          echo "Training ${train_folder}..."
          experiment_name="${dataset}${train_instance}_${model_type}"
          expfolder="experiments/${dataset}/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${dataset}${train_instance}
          echo "Using training model ${model_path}..."
          lr=0.001
          
          python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --framework keras --batcher legacy --model_type $model_type --loss per_class_margin | tee ${expfolder}/training.log
          
          #python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --model_type cnn224x224_autoencoder_almostoptim | tee ${expfolder}/training.log
          #python transcriber_trainer_workflow.py "${model_path}" "${train_folder}" train |  tee ${expfolder}/training_log.txt || exit

    done
done
done
