base=$1


# NEW! Trying out different models...

for dataset in Barrett; do # Iain Zhihan; do
    base_folder=/home/sethstewart/new_workspace/fcn-page-segmentation/data/
    for train_instance in A; do # B C; do
        for model_type in densenet_tiramisu; do
          train_folder="${base_folder}/${dataset}/$train_instance"
          #for block_layers in 2 3 4; do
          #for layers_per_block in 2 3 4; do
          #for feature_growth_rate in 4 8 16; do
          #for initial_features_per_block in 16 64; do
          #if [ $feature_growth_rate -ne 8 ] || [ $initial_features_per_block -ne 16 ]; then
          #if [ $block_layers -ne 4 ] || [ $layers_per_block -ne 4 ]; then
          #continue
          #fi
          #fi
          
          for block_layers in 5 4 2 3; do
          for layers_per_block in 6 4 2; do
          for feature_growth_rate in 16 8; do
          for initial_features_per_block in 16; do
          bottleneck=True # TODO
                    
          echo "Training ${train_folder}..."
          experiment_name="${dataset}${train_instance}_${model_type}_${block_layers}_${layers_per_block}_${initial_features_per_block}_${feature_growth_rate}"
          expfolder="experiments/${dataset}/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${dataset}${train_instance}
          echo "Using training model ${model_path}..."
          lr=0.0001
          
          python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --framework keras --batcher legacy --model_type $model_type --loss mse --block_layers $block_layers --layers_per_block $layers_per_block --initial_features_per_block $initial_features_per_block --feature_growth_rate $feature_growth_rate | tee ${expfolder}/training.log
          if [ ${PIPESTATUS[0]} != 0 ]; then
          echo "One or more errors encountered during training. Check for stack traces or abnormal conditions in the above program output."
          exit
          fi
          
          # Now validate against the NOT dataset.
          #test_folder="${base_folder}/${dataset}/not_$train_instance"
          #python validate.py --training_folder ${train_folder} --validation_folder ${train_folder} --test_folder ${test_folder} --log_dir ${expfolder}/$dataset/not$train_instance/ --lr $lr --framework keras --batcher legacy --model_type $model_type --loss mse | tee ${expfolder}/$dataset/not$train_instance/validation.log
          
          #python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --model_type cnn224x224_autoencoder_almostoptim | tee ${expfolder}/training.log
          #python transcriber_trainer_workflow.py "${model_path}" "${train_folder}" train |  tee ${expfolder}/training_log.txt || exit
done
done
done
done
    done
done
done
