experimentsdir=$1
num_classes=$2

for dataset in Barrett; do
    base_folder=/home/ubuntu/workspace/fcn-page-segmentation/data/
    for train_instance in A; do
        for dir in `ls $experimentsdir`; do          
          train_folder="${base_folder}/${dataset}/$train_instance"
          expfolder="${experimentsdir}/${dir}"
          model_path=${expfolder}/model_checkpoint.h5
          out_dir=${expfolder}/$dataset/not$train_instance/
          mkdir -p $out_dir
          # Now validate against the NOT dataset.
          test_folder="${base_folder}/${dataset}/not$train_instance"
          echo "Validating model ${model_path} against data in folder ${test_folder}"
          python validate.py --training_folder ${train_folder} --validation_folder ${train_folder} --test_folder ${test_folder} --log_dir ${out_dir} --output_folder ${out_dir} --framework keras --load_model_path ${model_path} --num_classes $num_classes --predthresholds 0.5,0.5,0.5,0.5,0.5,1.0 | tee ${out_dir}/validation.log

          
          echo ""
          echo ""
          echo ""
          echo ""
          echo ""
          
          # Now validate against diverse documents.
          echo "VALIDATING SAME MODEL on Diverse Document Dataset..."
          diverse="DiverseDocuments-Refined"
          out_dir=${expfolder}/$diverse/
          mkdir -p ${out_dir}
          test_folder="${base_folder}/$diverse"
          python validate.py --training_folder ${train_folder} --validation_folder ${train_folder} --test_folder ${test_folder} --log_dir ${out_dir} --output_folder ${out_dir} --framework keras --load_model_path ${model_path} --num_classes $num_classes --predthresholds 0.5,0.5,0.5,0.5,0.5,1.0 | tee ${out_dir}/validation.log
          
    done
    
    # Now run this command to aggregate all of the validation results into a single table.
    #cat ${experimentsdir}/*/$dataset/not$train_instance/validation.log | grep -e "f1_score \[" -e "Averages for" -e "Evaluating"
    #cat ${experimentsdir}/*/$dataset/not$train_instance/validation.log | grep -e "Averages for entire directory"
    grep -R -e "Averages for entire directory" -e "Trainable params" -A 1 ${experimentsdir}/*/$dataset/not$train_instance/validation.log | grep -e f1 -e "Trainable params" | sed 's/validation.log/\t/' | sed 's/ \[/\t/'
done
done
