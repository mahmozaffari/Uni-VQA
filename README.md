# Uni-VQA

This repository contains the codes for the EMNLP submission titled "General Knowledge is Power: Uncertainty-Aware Integration of LLMs for Reliable, Accurate, and Cost-Effective Visual Question Answering".

# Instructions:

## Precomputed Data
The precomputed data necessary for running the proposed LLM-based inference and evaluation framework is provided in the following [Link](https://drive.google.com/drive/folders/1Y2G5mkW72wKCjib0kmVF3BjjfFNkRHcA?usp=drive_link). This includes:

- Outputs of each individual VQA model (task-specific): These are the predictions from various VQA models.
- Answer Candidates Folder: Contains the answer candidates for each VQA model, which are necessary for running answer-candidate augmented LLM inference.
- Examples Folder: Contains the in-context examples computed based on BLIP-embedding similarities.

## Run LLM-based Inference

To perform answer-candidate augmented LLM-based inference, execute the following command:

    python main.py --path_to_save_preds $base_dir/results/ \
               --train_answer_candidates_path $base_dir/answer_candidates/${model}_${method}_train.csv \
               --val_answer_candidates_path $base_dir/answer_candidates/${model}_${method}_val.csv \
               --test_answer_candidates_path $base_dir/answer_candidates/${model}_${method}_test.csv \
               --use_answer_candidates True --dataset vqa2 \
               --evaluation_set $split \
               --num_candidates k1 k2 k3

- to run on the test and validation splits set the $split to 'test', and 'val' respectively.
- specify the number of candidates with one or more integer numbers indicating the number of answer-candidates provided as in prompt.
- Add --part n/N to partition the test into N parts, and do inference on the nth part.

