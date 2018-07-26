d=/media/sethstewart/ram; 
#d="/media/sethstewart/My Passport/Organized/datasets/Page Segmentation/DivaDia";
for model_type in densenet_tiramisu unet build_model_functional_old template_matcher_single_hidden_layer build_simple_hourglass; do 
    for dataset in Parzival SaintGall Washington; do 
        size="" #_1_8
        python train.py --training_folder $d/${dataset}${size}/train --validation_folder $d/${dataset}${size}/validation --log_dir experiments/${dataset}${size}/$model_type/ --lr 0.001 --framework keras --batcher simple --model_save_path model_checkpoint.h5 --model_type $model_type --loss categorical_crossentropy --min_height_scale 0.9 --max_height_scale 1.1 --loss_weights byclass --epochs 10 --batch_size=15 --crop_size 128 | tee experiments/${dataset}${size}/$model_type/training.log
          if [ ${PIPESTATUS[0]} != 0 ]; then
            echo "One or more errors encountered during training. Check for stack traces or abnormal conditions in the above program output."
            exit
          fi
    done;
done
