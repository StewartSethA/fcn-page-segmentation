base=$1


# NEW! Trying out different models...

for dataset in Barrett; do # Iain Zhihan; do
    base_folder=/home/sethstewart/new_workspace/fcn-page-segmentation/data/
    for train_instance in A; do # B C; do
        for model_type in template_matcher_single_hidden_layer; do
          train_folder="${base_folder}/${dataset}/$train_instance"
          for block_layers in 1 2 3 4; do
          for initial_kernel_size in 27 9 3; do
          for initial_features_per_block in 100 400; do
          test_instance=notA
          
          echo "Validating ${train_folder}..."
          experiment_name="${dataset}${train_instance}_${model_type}_${block_layers}_${initial_kernel_size}_${initial_features_per_block}"
          expfolder="experiments/${dataset}/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${dataset}${train_instance}
          echo "Using training model ${model_path}..."
          
          #python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --framework keras --batcher legacy --model_type $model_type --loss mse | tee ${expfolder}/training.log

          mkdir -p ${expfolder}/$dataset/not$train_instance/
          # Now validate against the NOT dataset.
          test_folder="${base_folder}/${dataset}/not$train_instance"
          python validate.py --training_folder ${train_folder} --validation_folder ${train_folder} --test_folder ${test_folder} --log_dir ${expfolder}/$dataset/not$train_instance/ --output_folder ${expfolder}/$dataset/not$train_instance/  --framework keras --batcher legacy --model_type $model_type --load_model_path ${expfolder}/model_checkpoint.h5 --loss mse --num_classes 5 --predthresholds 0.5,0.5,0.5,0.5,0.5,1.0 | tee ${expfolder}/$dataset/not$train_instance/validation.log

          
          #python validate.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --framework keras --batcher legacy --model_type $model_type --loss mse --block_layers $block_layers --initial_kernel_size $initial_kernel_size --initial_features_per_block $initial_features_per_block | tee ${expfolder}/training.log
          if [ ${PIPESTATUS[0]} != 0 ]; then
          echo "One or more errors encountered during training. Check for stack traces or abnormal conditions in the above program output."
          exit
          fi
          
          echo ""
          echo ""
          echo ""
          echo ""
          echo ""
          
          # Now validate against diverse documents.
          echo "VALIDATING SAME MODEL on Diverse Document Dataset..."
          diverse="DiverseDocuments-Refined"
          mkdir -p ${expfolder}/$diverse/
          test_folder="${base_folder}/$diverse"
          python validate.py --training_folder ${train_folder} --validation_folder ${train_folder} --test_folder ${test_folder} --log_dir ${expfolder}/$diverse/ --output_folder ${expfolder}/$diverse/  --framework keras --batcher legacy --model_type $model_type --load_model_path ${expfolder}/model_checkpoint.h5 --num_classes 5 --predthresholds 0.5,0.5,0.5,0.5,0.5,1.0 | tee ${expfolder}/$diverse/validation.log
          if [ ${PIPESTATUS[0]} != 0 ]; then
          echo "One or more errors encountered during validation. Check for stack traces or abnormal conditions in the above program output."
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
