
    base_folder=/home/ubuntu/workspace/fcn-page-segmentation/data/
    for train_instance in BirthRecord; do
        for model_type in build_simple_hourglass densenet_tiramisu template_matcher_single_hidden_layer build_simple_hourglass unet build_model_functional; do
          train_folder="${base_folder}/$train_instance"

          echo "Training ${train_folder}..."
          experiment_name="${train_instance}_${model_type}"
          expfolder="experiments/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${train_instance}
          echo "Using training model ${model_path}..."
          lr=0.001
          
          python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --framework keras --batcher legacy --model_type $model_type --loss mse | tee ${expfolder}/training.log
          
          # --loss_weights byclass
          if [ ${PIPESTATUS[0]} != 0 ]; then
          echo "One or more errors encountered during training. Check for stack traces or abnormal conditions in the above program output."
          exit
          fi
          
          #python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --model_type cnn224x224_autoencoder_almostoptim | tee ${expfolder}/training.log
          #python transcriber_trainer_workflow.py "${model_path}" "${train_folder}" train |  tee ${expfolder}/training_log.txt || exit

    done
done
done
