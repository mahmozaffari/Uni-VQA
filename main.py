import torch
import numpy as np
import pandas as pd
from ast import literal_eval
import os, sys
import json
import random

from source.vq2_vqa_in_context_learning import val_in_context_learning_vqa2_answer_candidates

from config import get_config
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, BlipForImageTextRetrieval
from transformers import set_seed


#get confing variables 
cnf = get_config(sys.argv)
dataset_to_use = cnf.dataset
train_images_dir = cnf.train_images_dir
val_images_dir = cnf.val_images_dir
test_images_dir = cnf.test_images_dir

number_of_candidates = cnf.num_candidates
test_answer_candidates_path = cnf.test_answer_candidates_path
val_answer_candidates_path = cnf.val_answer_candidates_path
train_answer_candidates_path = cnf.train_answer_candidates_path

n_shots = cnf.n_shots
k_ensemble = cnf.k_ensemble
no_of_captions = cnf.no_of_captions
path_to_save_preds = cnf.path_to_save_preds # dir
assert(not os.path.isfile(path_to_save_preds))
if not os.path.exists(path_to_save_preds):
    os.makedirs(path_to_save_preds)

file_name = f'{cnf.dataset}_{cnf.evaluation_set}_nshots_{n_shots}_kensemble_{k_ensemble}_ncaptions_{no_of_captions}_with_candidates'

if cnf.part is not None:
    part_i = int(cnf.part.split('/')[0])
    part_N = int(cnf.part.split('/')[1])
    assert(part_i <= part_N)
    assert(part_i > 0)
    file_name = f'{file_name}_part_{part_i}_{part_N}.csv'
    path_to_save_preds = os.path.join(path_to_save_preds,file_name)
else:
    part_i = None
    part_N = None
    file_name = f'{file_name}.csv'
    path_to_save_preds = os.path.join(path_to_save_preds,file_name)


#set up device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load Llama model
llama_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
llama_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_model = llama_model.to(device, dtype=torch.float16)

print('loaded models')
'''
#load annotations
if cnf.train_annotations_path[-3:] == "npy":
    train_annotations = np.load(cnf.train_annotations_path, allow_pickle=True)
    val_annotations = np.load(cnf.val_annotations_path, allow_pickle=True)
    test_annotations = np.load(cnf.test_annotations_path, allow_pickle=True)
    train_annotations_df = pd.DataFrame(list(train_annotations[1:]))
    val_annotations_df = pd.DataFrame(list(val_annotations[1:]))
    test_annotations_df = pd.DataFrame(list(test_annotations[1:]))
else:
    Warning('Not supported yet')
    quit()
    train_annotations_df = pd.read_csv(cnf.train_annotations_path)
    val_annotations_df = pd.read_csv(cnf.val_annotations_path)
    test_annotations_df = pd.read_csv(cnf.test_annotations_path)

print('loaded annotations')

# Load Embeddings
train_q_embedds = pd.read_csv(cnf.blip_train_question_embedds_path)
train_i_embedds = pd.read_csv(cnf.blip_train_image_embedds_path)
if cnf.debug:
    train_q_embedds = train_q_embedds[:100]
    train_i_embedds = train_i_embedds[:100]
train_q_embedds.question_embedd = train_q_embedds.question_embedd.apply(json.loads)
train_i_embedds.image_embedd = train_i_embedds.image_embedd.apply(json.loads)

val_q_embedds = pd.read_csv(cnf.blip_val_question_embedds_path)
val_i_embedds = pd.read_csv(cnf.blip_val_image_embedds_path)
if cnf.debug:
    val_q_embedds = val_q_embedds[:100]
    val_i_embedds = val_i_embedds[:100]
val_q_embedds.question_embedd = val_q_embedds.question_embedd.apply(json.loads)
val_i_embedds.image_embedd = val_i_embedds.image_embedd.apply(json.loads)
print('loaded embeddings')

# Load Captions
train_captions = pd.read_csv(cnf.train_captions_path)
if cnf.debug:
    train_captions = train_captions[:100]
train_captions.captions = train_captions.captions.apply(literal_eval)

val_captions = pd.read_csv(cnf.val_captions_path)
if cnf.debug:
    val_captions = val_captions[:100]
val_captions.captions = val_captions.captions.apply(literal_eval)

print('Loaded captions')

import pickle as pkl
with open('dataset/vqa2_data.pkl', 'wb') as f:
    data = pkl.dump({'train_annotations_df':train_annotations_df, 
                     'val_annotations_df': val_annotations_df,
                     'test_annotations_df': test_annotations_df,
                     'train_q_embedds': train_q_embedds,
                     'train_i_embedds': train_i_embedds,
                     'val_q_embedds': val_q_embedds, 
                     'val_i_embedds': val_i_embedds, 
                     'train_captions': train_captions,
                     'val_captions': val_captions}, f)
'''

import pickle as pkl
with open('dataset/vqa2_data.pkl', 'rb') as f:
    data = pkl.load(f)
    train_annotations_df = data['train_annotations_df']
    if cnf.evaluation_set == "val":
        val_annotations_df = data['val_annotations_df']
    else:
        val_annotations_df = data['test_annotations_df']
    train_q_embedds = data['train_q_embedds']
    train_i_embedds = data['train_i_embedds']
    val_q_embedds = data['val_q_embedds']
    val_i_embedds = data['val_i_embedds']
    train_captions = data['train_captions']
    val_captions = data['val_captions']

if cnf.examples_path is not None:
    context_examples_df = pd.read_csv(cnf.examples_path)
    context_examples_df.examples = context_examples_df.examples.apply(literal_eval)
else:
    context_examples_df = None
train_candidates = pd.read_csv(train_answer_candidates_path)
if cnf.evaluation_set == "val":
    val_candidates = pd.read_csv(val_answer_candidates_path)
else:
    val_candidates = pd.read_csv(test_answer_candidates_path)

val_candidates.top_k_answers = val_candidates.top_k_answers.apply(literal_eval)
val_candidates.top_k_confidences = val_candidates.top_k_confidences.apply(literal_eval)
train_candidates.top_k_answers = train_candidates.top_k_answers.apply(literal_eval)
train_candidates.top_k_confidences = train_candidates.top_k_confidences.apply(literal_eval)

#sort the questions according to the ascending order of first entry in top-k-confidences
val_candidates['first_confidence'] = val_candidates.top_k_confidences.apply(lambda x: x[0])
val_candidates_sorted = val_candidates.sort_values(by='first_confidence', ascending=True)


if part_i is not None:
    val_annotations_df = val_annotations_df.iloc[part_i-1::part_N]
    print('partitioned into {} parts, using part {}'.format(part_N, part_i))
    print('val_annotations_df.shape', val_annotations_df.shape)

if __name__ == "__main__":
    if dataset_to_use == "vqa2":
        train_annotations_df.answers = train_annotations_df.answers
        val_annotations_df.answers = val_annotations_df.answers

        results_df = val_in_context_learning_vqa2_answer_candidates(llama_model=llama_model, 
                                                        llama_tokenizer=llama_tokenizer, 
                                                        train_annotations_df=train_annotations_df, 
                                                        val_annotations_df=val_annotations_df,
                                                        train_q_embedds=train_q_embedds, 
                                                        train_i_embedds=train_i_embedds,
                                                        val_q_embedds=val_q_embedds, 
                                                        val_i_embedds=val_i_embedds,
                                                        train_captions=train_captions, 
                                                        val_captions=val_captions, 
                                                        val_candidates=val_candidates,
                                                        train_candidates=train_candidates,
                                                        num_candidates=number_of_candidates,
                                                        train_images_dir=train_images_dir, 
                                                        val_images_dir=val_images_dir, 
                                                        n_shots=n_shots,k_ensemble=k_ensemble,
                                                        MAX_CAPTION_LEN=30,
                                                        NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions, 
                                                        path_to_save_preds=path_to_save_preds,
                                                        device=device,
                                                        context_examples_df=context_examples_df,
                                                        part_i=part_i,
                                                        part_N=part_N)