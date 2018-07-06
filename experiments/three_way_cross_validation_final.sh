for dataset in Barrett; do # Iain Zhihan; do
    base_folder=/home/sethstewart/new_workspace/fcn-page-segmentation/data/
    for train_instance in A; do # B; do # B C; do
        for model_type in build_model_functional; do #unet densenet_tiramisu template_matcher_single_hidden_layer build_simple_hourglass build_model_functional; do
          train_folder="${base_folder}/${dataset}/$train_instance"

          echo "Validating ${train_folder}..."
          experiment_name="${dataset}${train_instance}_${model_type}"
          expfolder="experiments/${dataset}/${experiment_name}"
          mkdir -p $expfolder
          model_path=${expfolder}/${dataset}${train_instance}
          echo "Using training model ${model_path}..."
          lr=0.0001
          
          #python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --framework keras --batcher legacy --model_type $model_type --loss mse | tee ${expfolder}/training.log

          mkdir -p ${expfolder}/$dataset/not$train_instance/
          # Now validate against the NOT dataset.
          test_folder="${base_folder}/${dataset}/not$train_instance"
          python validate.py --training_folder ${train_folder} --validation_folder ${train_folder} --test_folder ${test_folder} --log_dir ${expfolder}/$dataset/not$train_instance/ --output_folder ${expfolder}/$dataset/not$train_instance/  --lr $lr --framework keras --batcher legacy --model_type $model_type --load_model_path ${expfolder}/model_checkpoint.h5 --loss mse --num_classes 6 --predthresholds 0.5,0.5,0.5,0.5,0.5,1.0 | tee ${expfolder}/$dataset/not$train_instance/validation.log
          
          if [ ${PIPESTATUS[0]} != 0 ]; then
          echo "One or more errors encountered during validation. Check for stack traces or abnormal conditions in the above program output."
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
          python validate.py --training_folder ${train_folder} --validation_folder ${train_folder} --test_folder ${test_folder} --log_dir ${expfolder}/$diverse/ --output_folder ${expfolder}/$diverse/ --lr $lr --framework keras --batcher legacy --model_type $model_type --load_model_path ${expfolder}/model_checkpoint.h5 --loss mse --num_classes 6 --predthresholds 0.5,0.5,0.5,0.5,0.5,1.0 | tee ${expfolder}/$diverse/validation.log
          if [ ${PIPESTATUS[0]} != 0 ]; then
          echo "One or more errors encountered during validation. Check for stack traces or abnormal conditions in the above program output."
          exit
          fi
          #python train.py --training_folder ${train_folder} --validation_folder ${train_folder} --log_dir ${expfolder} --lr $lr --model_type cnn224x224_autoencoder_almostoptim | tee ${expfolder}/training.log
          #python transcriber_trainer_workflow.py "${model_path}" "${train_folder}" train |  tee ${expfolder}/training_log.txt || exit

    done
done
done
