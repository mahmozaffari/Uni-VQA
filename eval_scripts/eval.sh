
val_file=$1     # val_predict.json for VQA v2
out_dir=$2
k=$3
shift 3
acc_targets="$@"

python eval_scripts/run.py \
-q $data_dir/v2_OpenEnded_mscoco_val2014_questions.json \
-a $data_dir/v2_mscoco_val2014_annotations.json \
-p $val_file \
-o $out_dir \
-l $k \
