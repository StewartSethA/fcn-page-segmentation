base=$1



      batcher_type="legacy"
      for model_type in "densenet" "hourglass"; do
        for feature_growth_rate in 4; do
          for initial_features_per_block in 20 16; do
            for block_layers in 2 4; do
                for layers_per_block in 3; do
                    for bottleneck_feats in 16 20; do
                        for bottleneck_growth_rate in 4; do

                          for dataset in Barrett Iain Zhihan; do
                              base_folder=/home/ubuntu/workspace/document-layout-analysis/data/Forms/
                              for train_instance in A B C; do
                                train_folder="${base_folder}/${dataset}/$train_instance"
                                validation_folder="${base_folder}/${dataset}/$train_instance"

                          experiment_params="${model_type}-in${initial_features_per_block}-gr${feature_growth_rate}-bl${block_layers}-lb${layers_per_block}-bf${bottleneck_feats}-bgr${bottleneck_growth_rate}"
                          echo "Training and validating ${train_folder}, ${validation_folder}"
                          expfolder="experiments/${dataset}/${experiment_params}"
                          mkdir -p $expfolder
                          python train.py "${train_folder}" "${validation_folder}" --block_layers $block_layers --layers_per_block $layers_per_block --initial_features_per_block $initial_features_per_block --feature_growth_rate $feature_growth_rate --bottleneck_feats $bottleneck_feats --bottleneck_growth_rate $bottleneck_growth_rate --model_type ${model_type} --batcher ${batcher_type}|  tee ${expfolder}/training_log.txt || exit

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
done
