from argparse import ArgumentParser
import os

base_path='/path/to/dir'
coco_path='/path/to/coco/dir'

def get_config(sysv):
    parser = ArgumentParser(description='In-context learning variables.')
    parser.add_argument('--dataset', type=str, choices = ['vqa2'], default='vqa2', help='dataset to use')
    parser.add_argument('--evaluation_set', type=str, choices = ['val', 'test'], default='val', help='Set to perform the experiments')
    parser.add_argument('--train_annotations_path', type=str, default=f'{base_path}]/reliablevqa/datasets/vqa2/defaults/annotations/imdb_train2014.npy', help='The path to the train annotations csv file')
    parser.add_argument('--val_annotations_path', type=str, default=f'{base_path}/reliablevqa/datasets/vqa2/reliable_vqa/annotations/imdb_val2014-dev.npy', help='The path to the val annotations csv file')
    parser.add_argument('--test_annotations_path', type=str, default=f'{base_path}/reliablevqa/datasets/vqa2/reliable_vqa/annotations/imdb_val2014-test.npy', help='The path to the test annotations csv file')

    parser.add_argument('--train_images_dir', type=str, default=f'{coco_path}/coco/train2014', help='Path for the training images dir')
    parser.add_argument('--val_images_dir', type=str, default=f'{coco_path}/coco/val2014', help='Path for the val images dir')
    parser.add_argument('--test_images_dir', type=str, default=f'{coco_path}/coco/val2014', help='Path for the test images dir (only for A-OK-VQA)')

    parser.add_argument('--n_shots', type=int, default=10, help='Number of shots for in-context-learning')
    parser.add_argument('--k_ensemble', type=int, default=5, help='Number of ensmembles for in-context-learning')
    parser.add_argument('--no_of_captions', type=int, default=9, help='Number of question informative captions for in-context-learning')
    parser.add_argument('--use_mcan_examples', type=str, default="False", choices=["True", "False"], help='If true uses the mcan based shot selection strategy. If false uses the avg question and image similarity')
    parser.add_argument('--mcan_examples_path', type=str, default=None, help='The path to the json file containing the mcan examples')
    parser.add_argument('--use_answer_candidates', type=str, default="False", choices=["True", "False"], help='If true uses the answer candidates for in-context-learning')
    parser.add_argument('--test_answer_candidates_path', type=str, default=None, help='The path to the json file containing the answer candidates')
    parser.add_argument('--val_answer_candidates_path', type=str, default=None, help='The path to the json file containing the answer candidates')
    parser.add_argument('--train_answer_candidates_path', type=str, default=None, help='The path to the json file containing the answer candidates')
    parser.add_argument('--llama_path', type=str, default=None, help='The path to the llama (1 or 2) weights')
    parser.add_argument('--blip_train_question_embedds_path', type=str, default='./blip_embedds/vqa2/normalized_blip_train_embeds_base_txt.csv', help='The path to the normalized blip train question embeddings')
    parser.add_argument('--blip_train_image_embedds_path', type=str, default='./blip_embedds/vqa2/normalized_blip_train_embeds_base_img.csv', help='The path to the normalized blip train image embeddings')
    parser.add_argument('--blip_val_question_embedds_path', type=str, default='./blip_embedds/vqa2/normalized_blip_val_embeds_base_txt.csv', help='The path to the normalized blip val question embeddings')
    parser.add_argument('--blip_val_image_embedds_path', type=str, default='./blip_embedds/vqa2/normalized_blip_val_embeds_base_img.csv', help='The path to the normalized blip val image embeddings')
    parser.add_argument('--blip_test_question_embedds_path', type=str, default=None, help='The path to the normalized blip test question embeddings (only for A-OK-VQA)')
    parser.add_argument('--blip_test_image_embedds_path', type=str, default=None, help='The path to the normalized blip test image embeddings (only for A-OK-VQA)')

    parser.add_argument('--train_captions_path', type=str, default='./question_related_captions/vqa2/v2_OpenEnded_mscoco_train2014_questions_pnp_3b_captions_9_sorted.csv', help='The path to the train question informative captions')
    parser.add_argument('--val_captions_path', type=str, default='./question_related_captions/vqa2/v2_OpenEnded_mscoco_val2014_questions_pnp_3b_captions_9_sorted.csv', help='The path to the val question informative captions')
    parser.add_argument('--test_captions_path', type=str, default=None, help='The path to the train question informative captions (only for A-OK-VQA)')

    parser.add_argument('--path_to_save_preds', type=str, default=None, help='Path to save the final predictions (needs to have .csv extension)')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode.')
    parser.add_argument('--part', type=str, default=None, help='The part of the dataset to use (i/N)')
    parser.add_argument('--examples_path', type=str, default='./examples/vqa2/examples_top_100_.csv', help='The path to the json file containing the examples')
    parser.add_argument('--num_candidates', type=int, nargs='+')
    args, _ = parser.parse_known_args(sysv)

    return args 

