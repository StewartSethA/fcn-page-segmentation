base=$1


# NEW! Trying out different models...

for dataset in Barrett; do
    base_folder=`pwd`/data/
    for train_instance in A; do #B C A; do
        for model_type in build_simple_hourglass; do # densenet_tiramisu template_matcher_single_hidden_layer unet build_model_functional build_model_functional_old; do
          for block_layers in 1 2 3; do
          for layers_per_block in 1 2; do
          for initial_features_per_block in 16 32; do
          for initial_kernel_size in 3 9; do
          if [ $initial_kernel_size -ne 3 ] && [ $block_layers -ne 3 ]; then
          continue
          fi
          #for block_layers in 2 4; do
          #for layers_per_block in 1 2; do
          #if [ $block_layers -ne 4 ] && [ $layers_per_block -ne 1 ]; then
          #continue
          #fi
          #for initial_features_per_block in 16 32; do
	  #if [ $block_layers -ne 4 ] && [ $initial_features_per_block -eq 16 ]; then
          #continue
	  #fi
          #if [ $initial_features_per_block -ne 32 ] && [ $layers_per_block -eq 2 ]; then
          #continue
          #fi
          #for initial_kernel_size in 3 9; do
          #if [ $block_layers -ne 4 ] && [ $initial_kernel_size -ne 9 ]; then
          #continue
          #fi
          for feature_growth_rate in 8 16; do
          #if [ $block_layers -ne 4 ] && [ $feature_growth_rate -ne 16 ]; then
          #continue
          #fi
          for feature_growth_type in add multiply; do
          if [ $feature_growth_type == 'multiply' ]; then
          if [ $feature_growth_rate -ne 16 ]; then
          continue
          fi
          feature_growth_rate=2
          fi
          if [ $block_layers -ne 4 ] && [ $feature_growth_type != 'add' ]; then
          continue
          fi
          train_folder="${base_folder}/${dataset}/$train_instance"

          echo "Training ${train_folder}..."
          experiment_name="${dataset}${train_instance}_${model_type}_${block_layers}_${layers_per_block}_${initial_features_per_block}_${initial_kernel_size}_${feature_growth_rate}_${feature_growth_type}"
          expfolder="experiments/${dataset}/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${dataset}${train_instance}
          echo "Using training model ${model_path}..."
          #lr=0.001
          lr=0.0002
          
          python train.py --epochs 80 --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --framework keras --batcher legacy --model_type $model_type --loss mse --block_layers $block_layers --layers_per_block $layers_per_block --initial_features_per_block $initial_features_per_block --initial_kernel_size $initial_kernel_size --feature_growth_rate $feature_growth_rate --feature_growth_type $feature_growth_type | tee ${expfolder}/training.log
          
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
    done
done
done
