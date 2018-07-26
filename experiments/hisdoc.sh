d=/media/sethstewart/ram/HisDB; 
#d="/media/sethstewart/My Passport/Organized/datasets/Page Segmentation/DivaDia";
for scale in 13 25 50 100; do
for model_type in densenet_tiramisu unet build_model_functional_old template_matcher_single_hidden_layer build_simple_hourglass; do 
    for dataset in CB55 CS18 CS863; do 
        size="" #_1_8
        mkdir -p experiments/${dataset}${size}/$model_type/
        python train.py --training_folder $d/scale_${scale}/$dataset/training --validation_folder $d/scale_${scale}/$dataset/validation --log_dir experiments/${dataset}_${scale}/$model_type/ --lr 0.001 --framework keras --batcher simple --model_save_path model_checkpoint.h5 --model_type $model_type --loss categorical_crossentropy --min_height_scale 0.9 --max_height_scale 1.1 --loss_weights byclass --epochs 20 --batch_size=15 --crop_size 256 | tee experiments/${dataset}${size}/$model_type/training.log
          if [ ${PIPESTATUS[0]} != 0 ]; then
            echo "One or more errors encountered during training. Check for stack traces or abnormal conditions in the above program output."
            exit
          fi
    done;
done;
done
