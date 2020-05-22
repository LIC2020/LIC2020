#!/usr/bin/env bash
export DATA_PATH=./xgb_data2
export MODEL_PATH=./model14
nohup python xgb_train.py \
    --model_path $MODEL_PATH \
    --train_data_file $DATA_PATH/train_cands.csv \
    --valid_data_file $DATA_PATH/valid_cands.csv \
    --test_data_file $DATA_PATH/test1_cands.csv \
    --x_train_file $DATA_PATH/x_train.csv \
    --x_valid_file $DATA_PATH/x_valid.csv \
    --x_test_file $DATA_PATH/x_test.csv \
    --y_train_file $DATA_PATH/y_train.csv \
    --y_valid_file $DATA_PATH/y_valid.csv \
    --y_test_file $DATA_PATH/y_test.csv \
    --pred_data_file $MODEL_PATH/results_xgb.csv \
    --num_boost_round 200 \
    --early_stopping_rounds 10 \
    --eta 0.1 \
    --gamma 0 \
    --max_depth 3 \
    --min_child_weight 5 \
    --max_delta_step 0 \
    --subsample 0.8 \
    --colsample_bytree 0.9 \
    --colsample_bylevel 0.8 \
    --lamda 1 \
    --alpha 0 \
    --scale_pos_weight 1 \
    > ./log_dir/14_train.log 2>&1 &
#
#nohup python xgb_train.py \
#    --model_path $MODEL_PATH \
#    --train_data_file $DATA_PATH/train_cands.csv \
#    --valid_data_file $DATA_PATH/train_cands.csv \
#    --test_data_file $DATA_PATH/test1_cands.csv \
#    --x_train_file $DATA_PATH/x_train_add_leijia_prob_ziwei.csv \
#    --x_valid_file $DATA_PATH/x_train_add_leijia_prob_ziwei.csv \
#    --x_test_file $DATA_PATH/x_test_add_leijia_prob_ziwei.csv \
#    --y_train_file $DATA_PATH/y_train_add_leijia_prob_ziwei.csv \
#    --y_valid_file $DATA_PATH/y_train_add_leijia_prob_ziwei.csv\
#    --y_test_file $DATA_PATH/y_test_add_leijia_prob_ziwei.csv \
#    --pred_data_file $MODEL_PATH/results_xgb.csv \
#    --num_boost_round 200 \
#    --early_stopping_rounds 100 \
#    --eta 0.3 \
#    --gamma 0 \
#    --max_depth 3 \
#    --min_child_weight 5 \
#    --max_delta_step 0 \
#    --subsample 1 \
#    --colsample_bytree 1 \
#    --colsample_bylevel 1 \
#    --lamda 1 \
#    --alpha 0 \
#    > ./log_dir/31_train.log 2>&1 &