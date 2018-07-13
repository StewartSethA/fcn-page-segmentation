base=$1


# NEW! Trying out different models...

for dataset in Barrett; do
    base_folder=`pwd`/data/
    for train_instance in A; do #B C A; do
        for model_type in unet; do # densenet_tiramisu template_matcher_single_hidden_layer unet build_model_functional build_model_functional_old; do
          for block_layers in 4 5 6 3 2; do #2 3 4 5; do
          for layers_per_block in 2 1 3; do
          for initial_features_per_block in 32 64 16; do # 16 32 64; do
          if [ $block_layers -ge 5 ] && [ $initial_features_per_block -gt 32 ]; then
          continue
          fi
	  #if [ $block_layers -ne 4 ] && [ $initial_features_per_block -ne 16 ]; then
      #    continue
	  #fi
          #if [ $initial_features_per_block -ne 64 ] && [ $layers_per_block -ne 2 ]; then
          #continue
          #fi
          train_folder="${base_folder}/${dataset}/$train_instance"

          echo "Training ${train_folder}..."
          experiment_name="${dataset}${train_instance}_${model_type}_${block_layers}_${layers_per_block}_${initial_features_per_block}"
          expfolder="experiments/${dataset}/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${dataset}${train_instance}
          echo "Using training model ${model_path}..."
          #lr=0.001
          lr=0.0002

          python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --epochs 80 --log_dir ${expfolder} --lr $lr --framework keras --batcher legacy --model_type $model_type --loss mse --block_layers $block_layers --layers_per_block $layers_per_block --initial_features_per_block $initial_features_per_block | tee ${expfolder}/training.log

          # --loss_weights byclass
          #if [ ${PIPESTATUS[0]} != 0 ]; then
          #echo "One or more errors encountered during training. Check for stack traces or abnormal conditions in the above program output."
          #exit
          #fi

          #python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --model_type cnn224x224_autoencoder_almostoptim | tee ${expfolder}/training.log
          #python transcriber_trainer_workflow.py "${model_path}" "${train_folder}" train |  tee ${expfolder}/training_log.txt || exit
	  done
	  done
          done
    done
done
done
