# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))

from reliable_vqa_eval import ReliabilityEval
from vqa import VQA
import matplotlib.pyplot as plt


def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data


def load_data(
    ques_file, ann_file, res_file, llm_res_file
):
    questions = load_json(ques_file)
    annotations = load_json(ann_file)

    ann_vqa = VQA(annotations=annotations, questions=questions)
    all_qids = ann_vqa.getQuesIds()

    vqa_eval = ReliabilityEval(
        all_qids, n=2
    )
    res_vqa = ann_vqa.loadRes(VQA(), res_file)
    if llm_res_file is not None:
        llm_res_vqa = ann_vqa.loadRes(VQA(), llm_res_file)
    else:
        llm_res_vqa = None


    return ann_vqa, res_vqa, llm_res_vqa, vqa_eval


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run reliable VQA evaluations.")
    parser.add_argument(
        "-q", "--questions", required=True, help="Path to question json file"
    )
    parser.add_argument(
        "-a", "--annotations", required=True, help="Path to annotation json file"
    )
    parser.add_argument(
        "-p", "--predictions", required=True, help="Path to prediction json file"
    )
    parser.add_argument(
        "-l", "--llm_predictions", required=False, help="Path to llm prediction json file"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output directory for saving results and plots",
    )
    # To measure the delegation percentage at certain accuracy thresholds
    parser.add_argument(
        "--acc_targets",
        nargs="+",
        type=float,
        default=[69.09],  # Default Mistral-7b accuracy on test
        help="Accuracy thresholds for measuring delegation percentage",
    )

    return parser.parse_args()


def main(args):
    full_ques_file = args.questions
    full_ann_file = args.annotations
    result_file = args.predictions
    llm_result_file = args.llm_predictions


    if args.output == "" or args.output is None:
        dir = os.path.dirname(result_file)
    else:
        dir = args.output

    name=os.path.basename(result_file).split('.')[0]
    if not os.path.exists(dir):
        os.makedirs(dir)

    assert(os.path.exists(full_ques_file))
    assert(os.path.exists(full_ann_file))

    assert(os.path.exists(result_file))
    if llm_result_file is not None:
        assert(os.path.exists(llm_result_file))

    gt_data, pred_data, llm_pred_data, evaluator, = load_data(
        full_ques_file,
        full_ann_file,
        result_file,
        llm_result_file,
    )

    qids = set(pred_data.getQuesIds())
    
    evaluator.evaluate(
        gt_data,
        pred_data,
        llm_pred_data,
        quesIds=qids,
        dir=dir,
        name=name,
        acc_targets=args.acc_targets,
    )

    str_ = ""
    str_+= str(evaluator.accuracy['vqa_accuracy']) + " "

    str_+= str(evaluator.accuracy['ece']) + " "
    str_+= str(evaluator.accuracy['overconfidence']) + " "
    str_+= '\n'
    
    print(str_)
    
    print("VQA Accuracy: ", f" {evaluator.accuracy['vqa_accuracy']}")
    if llm_pred_data is not None:
        print("llm VQA Accuracy: ", f" {evaluator.accuracy['llm_vqa_accuracy']}")
        keys = []
        delegate_accuracies = []
        delegate_thresholds = []
        delegate_percentages = []
        delegate_eces = []
        keys_ece = []

        for r in [0.0, 0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98,0.99,1.1]:
            if f"delegate_accuracy_{r}" in evaluator.accuracy:
                keys.append(r)
                print(f"Delegate Accuracy ({r}): ", f" ({evaluator.accuracy[f'delegate_accuracy_{r}']})")
                delegate_accuracies.append(evaluator.accuracy[f'delegate_accuracy_{r}'])
                delegate_thresholds.append(r)
                delegate_percentages.append(evaluator.accuracy[f'delegate_percentage_{r}'])
            if f"delegate_ece_{r}" in evaluator.accuracy:
                keys_ece.append(r)
                delegate_eces.append(evaluator.accuracy[f'delegate_ece_{r}'])
                print(f"Delegate ECE ({r}): ", f" ({evaluator.accuracy[f'delegate_ece_{r}']})")
        import pickle as pkl
        with open(os.path.join(dir,'delegate_by_threshold_performance.pkl'),'wb') as f:
            pkl.dump({'delegate_accuracies':delegate_accuracies,'delegate_thresholds':delegate_thresholds,'delegate_percentages':delegate_percentages},f)
        with open(os.path.join(dir,'delegate_by_threshold_performance.txt'),'w') as f:
            f.write(",".join([str(d) for d in delegate_accuracies]+[str(evaluator.accuracy['llm_vqa_accuracy'])]))
            f.write("\n")
            f.write(",".join([str(d) for d in delegate_thresholds]+["1.0"]))
            f.write("\n")
            f.write(",".join([str(d) for d in delegate_percentages]+["100"]))
            f.write("\n")
        
        with open(os.path.join(dir,'delegate_by_threshold_ece.txt'),'w') as f:
            for i in range(len(delegate_eces)):
                f.write(f"{delegate_eces[i]}\n")

        f = plt.figure()
        plt.bar(keys,[evaluator.accuracy[f"delegate_accuracy_{k}"] for k in keys],color='b',width=0.05,alpha=0.5,edgecolor='b')
        plt.plot([0,1],[evaluator.accuracy['vqa_accuracy'],evaluator.accuracy['vqa_accuracy']],'r--',label='VQA Accuracy')
        plt.plot([0,1],[evaluator.accuracy['llm_vqa_accuracy'],evaluator.accuracy['llm_vqa_accuracy']],'g:',label='llm VQA Accuracy')
        plt.legend()

        plt.ylabel('Accuracy')
        plt.xlabel('Delegation Threshold')
        plt.ylim([evaluator.accuracy['vqa_accuracy']-1,max([evaluator.accuracy[f"delegate_accuracy_{k}"] for k in keys])+1])
        plt.savefig(os.path.join(dir,'delegate_accuracy.png'),bbox_inches='tight')
        plt.close()

        f = plt.figure()
        plt.plot(keys_ece,delegate_eces,'-ob')
        plt.plot([keys_ece[0],keys_ece[-1]],[evaluator.accuracy['ece']]*2,'--r')
        plt.title('ECE after Delegation')
        plt.ylabel('ECE')
        plt.xlabel('Delegation Threshold')
        plt.savefig(os.path.join(dir,'delegate_ece.png'),bbox_inches='tight')
        plt.close()

        delegate_ece_path=os.path.join(dir,'delegate_ece')
        #list files ending in .pkl
        delegate_ece_files = [f for f in os.listdir(delegate_ece_path) if f.endswith('.pkl')]
        delegate_ece_files.sort()
        delegate_ece_files = [os.path.join(delegate_ece_path,f) for f in delegate_ece_files]
        f = plt.figure(figsize=(8,4))
        plt.plot([0,1],[0,1],'--k',alpha=0.5,label='Perfect Calibration')
        numplots=0
        import numpy as np
        colors = plt.cm.tab10(np.linspace(0, 1, len(delegate_ece_files)))
        for i,f in enumerate(delegate_ece_files[::2]):
            numplots+=1
            #extract threshold form file name after ece_delegate_at_ and before _ece.pkl
            threshold = float(f.split('ece_delegate_at_')[1].split('_ece.pkl')[0])
            with open(f,'rb') as pkl_file:
                data = pkl.load(pkl_file)
                plt.plot(data['x_axis'],data['y_axis'],label=f"t:{round(threshold,2)},ece:{round(data['ece'],2)}",color=colors[i])
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
        plt.title('ECE after Delegation')
        plt.ylabel('ECE')
        plt.xlabel('Delegation Threshold')
        plt.savefig(os.path.join(dir,'delegate_ece_all.png'),bbox_inches='tight')


    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        json.dump(evaluator.accuracy, open(os.path.join(args.output, f"{name}_eval.json"), "w"))
        with open(os.path.join(args.output,f"{name}_eval.txt"), 'w') as f:
            f.write(str_)


if __name__ == "__main__":
    args = parse_arguments()

    print("\n\n----------")
    print("Arguments:")
    argvar_list = [arg for arg in vars(args)]
    for arg in argvar_list:
        print("\t{}: {}".format(arg, getattr(args, arg)))
    print("----------\n")

    main(args)
