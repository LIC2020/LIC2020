#!/usr/bin/env bash
export DATA_PATH=./xgb_data2
export MODEL_PATH=./model14
#nohup python xgb_predict.py \
#    --test_data_file $DATA_PATH/valid_cands.csv \
#    --x_test_file $DATA_PATH/x_valid.csv \
#    --y_test_file $DATA_PATH/y_valid.csv \
#    --pred_data_file $MODEL_PATH/results_xgb.csv_tmp_ \
#    --model_path $MODEL_PATH/model \
#    --ntree_limit 107 \
#    > ./log_dir/14_predict.log 2>&1 &

#nohup python xgb_predict.py \
#    --test_data_file $DATA_PATH/train_cands.csv \
#    --x_test_file $DATA_PATH/x_train_add_leijia_prob_ziwei.csv \
#    --y_test_file $DATA_PATH/y_train_add_leijia_prob_ziwei.csv \
#    --pred_data_file $MODEL_PATH/results_xgb.csv_tmp_ \
#    --model_path $MODEL_PATH/model \
#    --ntree_limit 51 \
#    > ./log_dir/31_predict.log 2>&1 &

python xgb_predict.py \
    --test_data_file $DATA_PATH/test1_cands.csv \
    --x_test_file $DATA_PATH/x_test.csv \
    --y_test_file $DATA_PATH/y_test.csv \
    --pred_data_file $MODEL_PATH/test1_results_xgb.csv_tmp_ \
    --model_path $MODEL_PATH/model \
    --ntree_limit 107

#nohup python xgb_predict.py \
#    --test_data_file $DATA_PATH/test1_cands.csv \
#    --x_test_file $DATA_PATH/x_test_add_leijia_prob_ziwei.csv \
#    --y_test_file $DATA_PATH/y_test_add_leijia_prob_ziwei.csv \
#    --pred_data_file $MODEL_PATH/test1_results_xgb.csv_tmp_ \
#    --model_path $MODEL_PATH/model \
#    --ntree_limit 40 \
#    > ./log_dir/29_predict.log 2>&1 &