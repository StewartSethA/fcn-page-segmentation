base=$1

for dataset in Barrett; do
    #base_folder='C:\Users\Seth\workspace\document-layout-analysis\data\Forms\' 
    base_folder=/home/ubuntu/workspace/document-layout-analysis/data/Forms/
    #/home/ubuntu/workspace/HBA-Refactor/HisDoc/data/scale_13
    train_folder="${base_folder}/${dataset}/training"
    validation_folder="${base_folder}/${dataset}/validation"

    
	
	
    for model_type in "hourglass"; do
    for feature_growth_rate in 4 8; do
    for initial_features_per_block in 4 32 16; do
            for block_layers in 1 2 4; do
                for layers_per_block in 1; do
                    for bottleneck_feats in 10; do
                        for bottleneck_growth_rate in 4; do
                            
                        experiment_params="${model_type}-in${initial_features_per_block}-gr${feature_growth_rate}-bl${block_layers}-lb${layers_per_block}-bf${bottleneck_feats}-bgr${bottleneck_growth_rate}"
                        echo "Training and validating ${train_folder}, ${validation_folder}"
                        expfolder="experiments/${dataset}/${experiment_params}"
                        mkdir -p $expfolder
                        python train.py "${train_folder}" "${validation_folder}" --block_layers $block_layers --layers_per_block $layers_per_block --initial_features_per_block $initial_features_per_block --feature_growth_rate $feature_growth_rate --bottleneck_feats $bottleneck_feats --bottleneck_growth_rate $bottleneck_growth_rate --model_type ${model_type} |  tee ${expfolder}/training_log.txt || exit
    
                        mv model_checkpoint.h5 "${expfolder}"/
                        cp loss_history.json "${expfolder}"/
                        cp training.png "${expfolder}"/
                        cp model.png "${expfolder}"/
                        cp LossMetrics.png "${expfolder}"/
                        mkdir -p ${expfolder}/validation
                        mv ${validation_folder}/*pred*.jpg "${expfolder}"/validation/
                        done
                    done
                done
            done
        done
    done
done
	
	for model_type in "densenet"; do
    for feature_growth_rate in 4 8; do
    for initial_features_per_block in 4 32 16; do
            for block_layers in 1 4; do
                for layers_per_block in 3; do
                    for bottleneck_feats in 12; do
                        for bottleneck_growth_rate in 4; do
                            
                        experiment_params="${model_type}-in${initial_features_per_block}-gr${feature_growth_rate}-bl${block_layers}-lb${layers_per_block}-bf${bottleneck_feats}-bgr${bottleneck_growth_rate}"
                        echo "Training and validating ${train_folder}, ${validation_folder}"
                        expfolder="experiments/${dataset}/${experiment_params}"
                        mkdir -p $expfolder
                        python train.py "${train_folder}" "${validation_folder}" --block_layers $block_layers --layers_per_block $layers_per_block --initial_features_per_block $initial_features_per_block --feature_growth_rate $feature_growth_rate --bottleneck_feats $bottleneck_feats --bottleneck_growth_rate $bottleneck_growth_rate --model_type ${model_type} |  tee ${expfolder}/training_log.txt || exit
    
                        mv model_checkpoint.h5 "${expfolder}"/
                        cp loss_history.json "${expfolder}"/
                        cp training.png "${expfolder}"/
                        cp model.png "${expfolder}"/
                        cp LossMetrics.png "${expfolder}"/
                        mkdir -p ${expfolder}/validation
                        mv ${validation_folder}/*pred*.jpg "${expfolder}"/validation/
                        done
                    done
                done
            done
        done
    done
done
	
done
