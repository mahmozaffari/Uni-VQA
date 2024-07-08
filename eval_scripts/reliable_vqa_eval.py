# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#### LICENSE from https://github.com/GT-Vision-Lab/VQA
# Copyright (c) 2014, Aishwarya Agrawal
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are
# those
# of the authors and should not be interpreted as representing official
# policies,
# either expressed or implied, of the FreeBSD Project.

import re

from collections import OrderedDict

import numpy as np
from sklearn.metrics import auc
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(?<=\d)(\,)+(?=\d)")
puncStrip = re.compile(
    r"(?<=[ \\;/\"`\[\](){}<>@=+_\-,?!])([\\;/\"`\[\](){}<>@=+_\-,?!])|([\\;/\"`\[\](){}<>@=+_\-,?!])(?=[ \\;/\"`\[\](){}<>@=+_\-,?!])"
)
puncStrip2 = re.compile(r"(?<=[a-zA-Z])([\\;/\"`\[\](){}<>@=+_\-,?!])(?=[a-zA-Z])")
puncStripBegin = re.compile(r"\A([ \\;/\"`\[\](){}<>@=+_\-,?!]+)(?=[a-zA-Z0-9 ])")
puncStripEnd = re.compile(r"(?<=[a-zA-Z0-9 ])([ \\;/\"`\[\](){}<>@=+_\-,?!]+)\Z")
spaceCleanup = re.compile(r"([ ]+)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


class ReliabilityEval:
    def __init__(self, quesIds, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = OrderedDict()
        self.evalThresholdQA = OrderedDict()
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.all_qids = quesIds

        self.fptp_stats = {}

    def evaluate(
        self,
        vqa,
        vqaRes,
        llm_vqaRes,
        quesIds=None,
        dir=None,
        name=None,
        acc_targets = None
    ):
        print('acc_targets:',acc_targets)
        if quesIds == None:
            quesIds = self.all_qids

        # =================================================
        # Compute accuracy
        # =================================================
        self.computeECE(vqa, vqaRes, quesIds=quesIds, dir=dir, name=name)
        
        
        if llm_vqaRes is not None:
            self.compute_llm_acc_vs_backbone_confidence(vqa, vqaRes, llm_vqaRes, quesIds=quesIds, dir=dir, name=name)
        
        self.computeAccuracy(vqa, vqaRes, llm_vqaRes, quesIds, is_threshold=False, dir=dir, name=name)

        if llm_vqaRes is not None:
            for delegate_threshold in [0.0,0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1+1e-9]:
                self.computeAccuracy_with_llm_delegate_in_lowconf(vqa, vqaRes, llm_vqaRes, quesIds=quesIds, is_threshold=False, dir=dir, delegate_threshold=delegate_threshold)
            
            self.computeAccuracy_with_llm_delegate_in_lowconf_by_percent(vqa, vqaRes, llm_vqaRes, quesIds=quesIds, is_threshold=False, dir=dir, acc_targets=acc_targets)

    def computeECE(self, vqa, vqaRes, quesIds=None,dir=None, name=None):
        accQA = []
        confs = []

        for quesId in quesIds:
            gt = vqa.qa[quesId]
            res = vqaRes.qa[quesId]
            if not 'multiple_choice_answer' in gt:
                answers = [a['answer'] for a in gt['answers']]
                # find multiple choice answer
                ans_counts = {}
                # Iterate through the list and count the occurrences of each element
                for ans in answers:
                    if ans in ans_counts:
                        ans_counts[ans] += 1
                    else:
                        ans_counts[ans] = 1
                gt['multiple_choice_answer'] = max(ans_counts, key=ans_counts.get)

            gt['multiple_choice_answer'] = gt['multiple_choice_answer'].replace("\n", " ")
            gt['multiple_choice_answer'] = gt['multiple_choice_answer'].replace("\t", " ")
            gt['multiple_choice_answer'] = gt['multiple_choice_answer'].strip()
            resAns = str(res['answer'])
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()

            try:
                resConf = res['confidence']
            except:
                resConf = res['probability']

            gtAcc = []
            gtAnswers = gt['multiple_choice_answer']

            # try with the processPunctuation and processDigitArticle later to see if it affects the accuracy
            acc = float(resAns == gtAnswers)
            accQA.append(acc)
            confs.append(resConf)
        
        ece = self._calc_ece_result(accQA, confs, dir, name)

        correct_idx = torch.Tensor(accQA).eq(1)
        underconfidence = (1-torch.Tensor(confs)[correct_idx]).mean().item()
        overconfidence = (torch.Tensor(confs)[~correct_idx]).mean().item()
        
        self.accuracy['ece'] = ece
        self.accuracy['overconfidence'] = overconfidence
        self.accuracy['underconfidence'] = underconfidence
        
    def compute_llm_acc_vs_backbone_confidence(self, vqa, vqaRes, llm_vqaRes, quesIds=None,dir=None, name=None):
        qids = []
        llm_accQA = [] # must be accuracy of llm (llmRes)
        accQA = [] # accuracy of backbone
        confs = [] # must be confidence of backbone (vqaRes)

        llm_notfound = 0
        for quesId in quesIds:
            # if quesId not in llm_vqaRes.qa:
            #     continue
            if quesId not in llm_vqaRes.qa:
                llm_notfound += 1
                continue

            gt = vqa.qa[quesId]
            res = vqaRes.qa[quesId]
            llm_res = llm_vqaRes.qa[quesId]
            if not 'multiple_choice_answer' in gt:
                answers = [a['answer'] for a in gt['answers']]
                # find multiple choice answer
                ans_counts = {}
                # Iterate through the list and count the occurrences of each element
                for ans in answers:
                    if ans in ans_counts:
                        ans_counts[ans] += 1
                    else:
                        ans_counts[ans] = 1
                gt['multiple_choice_answer'] = max(ans_counts, key=ans_counts.get)

            gt['multiple_choice_answer'] = gt['multiple_choice_answer'].replace("\n", " ")
            gt['multiple_choice_answer'] = gt['multiple_choice_answer'].replace("\t", " ")
            gt['multiple_choice_answer'] = gt['multiple_choice_answer'].strip()
            resAns = str(res['answer'])
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            llm_resAns = str(llm_res['llm_answer'])#str(llm_res['answer'])
            llm_resAns = llm_resAns.replace("\n", " ")
            llm_resAns = llm_resAns.replace("\t", " ")
            llm_resAns = llm_resAns.strip()
            
            try:
                resConf = res['confidence']
            except:
                resConf = res['probability']

            gtAcc = []
            gtAnswers = gt['multiple_choice_answer']

            # try with the processPunctuation and processDigitArticle later to see if it affects the accuracy
            llm_acc = float(llm_resAns == gtAnswers)
            acc = float(resAns == gtAnswers)
            llm_accQA.append(llm_acc)
            accQA.append(acc)
            confs.append(resConf)
            qids.append(quesId)
        print('llm found: ',len(llm_accQA))
        print(f"llm not found for {llm_notfound} questions")
        name=name+'_' if name is not None else ''
        with open(os.path.join(dir,name+'llm_acc_vs_backbone_confidence.csv'),'w') as f:
            for i in range(len(llm_accQA)):
                f.write(f"{qids[i]},{llm_accQA[i]},{accQA[i]},{confs[i]}\n")
        # similar to ece computation
        bin_boundaries = torch.linspace(0, 1, 10 + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        llam_accuracies = torch.Tensor(llm_accQA)
        accuracies = torch.Tensor(accQA)
        confidences = torch.Tensor(confs)
        x_axis = []
        y_axis_llm = []
        y_axis_backbone = []
        y_axis_llm_std = []
        y_axis_backbone_std = []
        x_axis_cnt = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                std, accuracy_in_bin = torch.std_mean(accuracies[in_bin].float())
                llm_std, llm_accuracy_in_bin = torch.std_mean(llam_accuracies[in_bin].float())
                avg_confidence_in_bin = confidences[in_bin].mean()
                x_axis_cnt.append(prop_in_bin.item())
                y_axis_llm_std.append(llm_std.item())
                y_axis_backbone_std.append(std.item())
                x_axis.append(avg_confidence_in_bin.item())
                y_axis_llm.append(llm_accuracy_in_bin.item())
                y_axis_backbone.append(accuracy_in_bin.item())

        y_axis_backbone = np.array(y_axis_backbone)
        y_axis_llm = np.array(y_axis_llm)
        y_axis_backbone_std = np.array(y_axis_backbone_std)
        y_axis_llm_std = np.array(y_axis_llm_std)
        f = plt.figure(figsize=(8,10))
        ax = plt.subplot(211)
        # plot line plot of avg accuracy vs confidences with std deviation
        ax.plot(x_axis, y_axis_llm, '-bo',label='llm')
        ax.plot(x_axis, y_axis_backbone, '-r*',label='backbone')

        plt.xlabel('confidences')
        plt.ylabel('accuracy')
        plt.legend()
        ax = plt.subplot(212)
        ax.bar(x_axis,x_axis_cnt,color='b',width=0.05,alpha=0.5,edgecolor='b')
        plt.xlabel('confidences')
        plt.ylabel('frequency')
        
        plt.savefig(os.path.join(dir, name+'llm_acc_vs_backbone_confidence.png'),bbox_inches='tight')
        with open(os.path.join(dir,name+'llm_acc_vs_backbone_confidence_bin2.txt'),'w') as f:
            f.write(','.join([str(x) for x in x_axis])+'\n')
            f.write(','.join([str(y) for y in y_axis_llm])+'\n')
        with open(os.path.join(dir,name+'backbone_acc_vs_backbone_confidence_bin.txt'),'w') as f:
            f.write(','.join([str(x) for x in x_axis])+'\n')
            f.write(','.join([str(y) for y in y_axis_backbone])+'\n')
        with open(os.path.join(dir,name+'llm_acc_vs_backbone_confidence_bin.txt'),'w') as f:
            for i in range(len(x_axis)):
                f.write(f"{x_axis[i]} {y_axis_llm[i]}\n")

    def _calc_ece_result(self, accuracies, confidences,dir,name):
        bin_boundaries = torch.linspace(0, 1, 10 + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = torch.Tensor(accuracies)
        confidences = torch.Tensor(confidences)
        ece = torch.zeros(1)
        ece_x_axis = []
        ece_y_axis = []
        ece_y_axis_std = []
        ece_x_axis_cnt = []
        
        #f = plt.figure()
        per_bin_acc_hist = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

            bins_, edges_ = np.histogram(accuracies[in_bin].float().numpy())
            per_bin_acc_hist.append((edges_,bins_))
            #plt.plot(edges_[:-1], bins_/sum(bins_),'-o',linewidth=2,label='confidence bin={}-{}'.format(round(bin_lower.item(),2),round(bin_upper.item(),2)))

            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                ece_x_axis_cnt.append(prop_in_bin.item())
            
                std, accuracy_in_bin = torch.std_mean(accuracies[in_bin].float())
                #accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_y_axis_std.append(std.item())
                ece_x_axis.append(avg_confidence_in_bin.item())
                ece_y_axis.append(accuracy_in_bin.item())
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        if name==None:
            name=''
        else:
            name+='_'
        if dir is not None:
            f = plt.figure()
            cmap = plt.get_cmap('viridis')
            width = (per_bin_acc_hist[0][0][1] - per_bin_acc_hist[0][0][0]) * 0.8
            for i, (edges, counts) in enumerate(per_bin_acc_hist):
                color = cmap(i / len(per_bin_acc_hist))
                plt.bar(edges[:-1] + i * width, counts, width=width, color=color, label=f'Conf. bin={bin_lowers[i]:.1f}-{bin_uppers[i]:.1f}')
            plt.xlabel('accuracy')
            plt.ylabel('Count')
            plt.title('Accuracy(0/1) Histogram in confidence bins')
            plt.legend()
            plt.xticks([0,1])
            plt.savefig(os.path.join(dir, name+'accuracy_distribution_in_confidence_bins.png'),bbox_inches='tight')
            
            f = plt.figure()
            for i, (edges, counts) in enumerate(per_bin_acc_hist):
                color = cmap(i / len(per_bin_acc_hist))
                plt.bar(edges[:-1] + i * width, counts/sum(counts), color=color, width=width, label=f'Conf. bin={bin_lowers[i]:.1f}-{bin_uppers[i]:.1f}')
            plt.xlabel('accuracy')
            plt.ylabel('Frequency')
            plt.title('Accuracy(0/1) distribution in confidence bins')
            plt.legend()
            plt.ylim([0,1.5])
            plt.xticks([0,1])
            plt.savefig(os.path.join(dir, name+'accuracy_distribution_in_confidence_bins_normalized.png'), bbox_inches='tight')
        if dir is not None:
            f = plt.figure()
            plt.bar(ece_x_axis,ece_x_axis,color='r',width=0.05,alpha=0.5,edgecolor='r')
            plt.bar(ece_x_axis,ece_y_axis,color='b',width=0.05,alpha=0.5,edgecolor='b',label='ECE={}'.format(round(ece.item(),4)))
            plt.plot([0,1],[0,1],color='k',linestyle='--',alpha=0.5)
            '''set xlabel and ylabel'''
            plt.xlabel('confidences')
            plt.ylabel('accuracy')
            plt.legend()
            plt.title('ECE plot')
            plt.savefig(os.path.join(dir, name+'eceplot.png'), bbox_inches='tight')
            print('ECE = {}'.format(ece.item()))

            f = plt.figure()
            plt.bar(ece_x_axis, ece_x_axis_cnt, color='b', width=0.05, alpha=0.5, edgecolor='b', label='ECE={}'.format(round(ece.item(), 4)))
            plt.xlabel('confidences')
            plt.ylabel('count')
            plt.title('ECE count plot')
            plt.savefig(os.path.join(dir, name+'count.png'), bbox_inches='tight')

            f = plt.figure()
            ''' plot that shows both mean and standard devatiaions for each bin'''
            #plt.bar(ece_x_axis, ece_y_axis, color='b', width=0.05, alpha=0.5, edgecolor='b', label='ECE={}'.format(round(ece_1.item(), 4)))
            plt.plot(ece_x_axis, ece_y_axis_std, '-o')
            '''set xlabel and ylabel'''
            plt.xlabel('confidences')
            plt.ylabel('accuracy')
            plt.title('ECE plot with standard deviation')
            plt.savefig(os.path.join(dir, name+'eceplot_std.png'), bbox_inches='tight')

        ece = ece.item()
        return ece

    def computeAccuracy_with_llm_delegate_in_lowconf(self, vqa, vqaRes, llm_vqaRes, quesIds=None, is_threshold=False, dir=None, delegate_threshold=0.5, name=None):
        accQA = []
        mc_accQA = []
        confidences = []
        percentage_by_llm = 0
        
        for quesId in quesIds:
            if quesId not in llm_vqaRes.qa:
                continue

            gt = vqa.qa[quesId]
            res = vqaRes.qa[quesId]
            llm_res = llm_vqaRes.qa[quesId]

            for ansDic in gt["answers"]:
                ansDic["answer"] = ansDic["answer"].replace("\n", " ")
                ansDic["answer"] = ansDic["answer"].replace("\t", " ")
                ansDic["answer"] = ansDic["answer"].strip()
            resAns = str(res["answer"])
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()

            llm_resAns = str(llm_res["llm_answer"]) #str(llm_res["answer"])
            llm_resAns = llm_resAns.replace("\n", " ")
            llm_resAns = llm_resAns.replace("\t", " ")
            llm_resAns = llm_resAns.strip()

            try:
                resConf = res["confidence"]
            except:
                resConf = res["probability"]
            confidences.append(resConf)

            gtAcc = []
            gtAnswers = [ans["answer"] for ans in gt["answers"]]
            gtAnswers_mc = gt['multiple_choice_answer']

            if len(set(gtAnswers)) > 1:
                for ansDic in gt["answers"]:
                    ansDic["answer"] = self.processPunctuation(ansDic["answer"])
                    ansDic["answer"] = self.processDigitArticle(ansDic["answer"])
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)

                llm_resAns = self.processPunctuation(llm_resAns)
                llm_resAns = self.processDigitArticle(llm_resAns)
            
            #######################################################
            if resConf < delegate_threshold:
                percentage_by_llm += 1
            for gtAnsDatum in gt["answers"]:
                otherGTAns = [item for item in gt["answers"] if item != gtAnsDatum]
                if resConf < delegate_threshold:
                    matchingAns = [item for item in otherGTAns if item["answer"] == llm_resAns]
                else:
                    matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            mc_acc = float(llm_resAns == gtAnswers_mc) if resConf < delegate_threshold else float(resAns == gtAnswers_mc)
            mc_accQA.append(mc_acc)
            #######################################################
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            risk = 1.0 - avgGTAcc
            accQA.append(avgGTAcc)
        percentage_by_llm = percentage_by_llm/len(accQA)
            ########################################################
        
        ece = self.delegate_ece(mc_accQA, confidences, dir=dir, name='ece_delegate_at_{}'.format(delegate_threshold))
        if not is_threshold:
            self.setllmDelegateAccuracy(accQA, delegate_threshold=delegate_threshold, delegate_percentage=percentage_by_llm,ece=ece)

    def computeAccuracy_with_llm_delegate_in_lowconf_by_percent(self, vqa, vqaRes, llm_vqaRes, quesIds=None, is_threshold=False, dir=None, name=None, acc_targets=None):
        accQA = []
        confidences = []
        llm_accQA = []
        
        for quesId in quesIds:
            if quesId not in llm_vqaRes.qa:
                continue

            gt = vqa.qa[quesId]
            res = vqaRes.qa[quesId]
            llm_res = llm_vqaRes.qa[quesId]

            for ansDic in gt["answers"]:
                ansDic["answer"] = ansDic["answer"].replace("\n", " ")
                ansDic["answer"] = ansDic["answer"].replace("\t", " ")
                ansDic["answer"] = ansDic["answer"].strip()
            resAns = str(res["answer"])
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()

            llm_resAns = str(llm_res["llm_answer"]) #str(llm_res["answer"])
            llm_resAns = llm_resAns.replace("\n", " ")
            llm_resAns = llm_resAns.replace("\t", " ")
            llm_resAns = llm_resAns.strip()

            try:
                resConf = res["confidence"]
            except:
                resConf = res["probability"]
            confidences.append(resConf)

            gtAcc = []
            gtAcc_llm = []
            gtAnswers = [ans["answer"] for ans in gt["answers"]]
            gtAnswers_mc = gt['multiple_choice_answer']

            if len(set(gtAnswers)) > 1:
                for ansDic in gt["answers"]:
                    ansDic["answer"] = self.processPunctuation(ansDic["answer"])
                    ansDic["answer"] = self.processDigitArticle(ansDic["answer"])
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)

                llm_resAns = self.processPunctuation(llm_resAns)
                llm_resAns = self.processDigitArticle(llm_resAns)
            
            #######################################################
            for gtAnsDatum in gt["answers"]:
                otherGTAns = [item for item in gt["answers"] if item != gtAnsDatum]
                matchingAns_llm = [item for item in otherGTAns if item["answer"] == llm_resAns]
                matchingAns_vqa = [item for item in otherGTAns if item["answer"] == resAns]
                acc_vqa = min(1, float(len(matchingAns_vqa)) / 3)
                acc_llm = min(1, float(len(matchingAns_llm)) / 3)
                gtAcc.append(acc_vqa)
                gtAcc_llm.append(acc_llm)
            #######################################################
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            avgGTAcc_llm = float(sum(gtAcc_llm)) / len(gtAcc_llm)

            accQA.append(avgGTAcc)
            llm_accQA.append(avgGTAcc_llm)
            ########################################################
        confidences, accQA, llm_accQA = zip(*sorted(zip(confidences, accQA, llm_accQA)))
        coverages_llm = []
        total_accuracies = []
        for i in range(1,len(confidences)+1):
            coverages_llm.append(i/len(confidences))
            llm_acc = sum(llm_accQA[:i])
            vqa_acc = sum(accQA[i:])
            total_accuracies.append((llm_acc+vqa_acc)/len(confidences))
        if dir is not None:
            f = plt.figure()
            plt.plot(coverages_llm, total_accuracies, '-', label='total accuracy')
            plt.xlabel('coverage by LLM')
            plt.ylabel('accuracy')
            plt.legend()
            plt.title('Total accuracy vs coverage')
            plt.savefig(os.path.join(dir, 'total_accuracy_vs_coverage.png'), bbox_inches='tight')
            plt.savefig(os.path.join(dir, 'total_accuracy_vs_coverage.eps'), format='eps')
            plt.close()
        # find accuracies at coverages 0.2 0.4 0.5 0.6 0.8 0.9
        coverage_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        accuracies_y = []
        for cov in coverage_x:
            idx = int(cov*len(confidences))
            accuracies_y.append(round(total_accuracies[idx]*100,2))
        
        if dir is not None:
            with open(os.path.join(dir, 'total_accuracy_vs_coverage.txt'), 'w') as f:
                f.write(','.join([str(x) for x in coverage_x])+'\n')
                f.write(','.join([str(y) for y in accuracies_y])+'\n')

        total_accuracies = [round(x*100,2) for x in total_accuracies]
        if acc_targets is not None:
            acc_at_target = []
            percentage_at_target = []
            for acc_target in acc_targets:
                idx = np.argmin(np.abs(np.array(total_accuracies)-acc_target))
                acc_at_target.append(round(total_accuracies[idx],2))
                percentage_at_target.append(round(coverages_llm[idx]*100,2))
            if dir is not None:
                with open(os.path.join(dir, 'coverage_at_accuracy_targets.txt'), 'w') as f:
                    f.write(','.join([str(x) for x in acc_targets])+'\n')
                    f.write(','.join([str(y) for y in acc_at_target])+'\n')
                    f.write(','.join([str(y) for y in percentage_at_target])+'\n')

    def delegate_ece(self, accuracies, confidences, dir=None, name=None):
        bin_boundaries = torch.linspace(0, 1, 10 + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = torch.Tensor(accuracies)
        confidences = torch.Tensor(confidences)
        ece = torch.zeros(1)
        ece_x_axis = []
        ece_y_axis = []
        ece_y_axis_std = []
        ece_x_axis_cnt = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                ece_x_axis_cnt.append(prop_in_bin.item())
            
                std, accuracy_in_bin = torch.std_mean(accuracies[in_bin].float())
                #accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_y_axis_std.append(std.item())
                ece_x_axis.append(avg_confidence_in_bin.item())
                ece_y_axis.append(accuracy_in_bin.item())
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        if dir is not None:
            import pickle as pkl
            dir = os.path.join(dir, 'delegate_ece')
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(os.path.join(dir, f'{name}_ece.pkl'), 'wb') as f:
                pkl.dump({'ece': ece.item(), 'x_axis': ece_x_axis, 'y_axis': ece_y_axis, 'x_axis_cnt': ece_x_axis_cnt}, f)

        ece = ece.item()
        return ece
    
    def computeAccuracy(self, vqa, vqaRes, llm_vqaRes, quesIds=None, is_threshold=False, dir=None, name=None):
        llm_performance = []
        accQA = []
        confsQA = []
        llm_accQA = []
        
        for quesId in quesIds:
            
            gt = vqa.qa[quesId]
            res = vqaRes.qa[quesId]
            try:
                llm_res = llm_vqaRes.qa[quesId]
                llm_found=True
            except:
                llm_found=False

            for ansDic in gt["answers"]:
                ansDic["answer"] = ansDic["answer"].replace("\n", " ")
                ansDic["answer"] = ansDic["answer"].replace("\t", " ")
                ansDic["answer"] = ansDic["answer"].strip()
            resAns = str(res["answer"])
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()

            if llm_found:
                llm_resAns = str(llm_res["llm_answer"]) #str(llm_res["answer"])
                llm_resAns = llm_resAns.replace("\n", " ")
                llm_resAns = llm_resAns.replace("\t", " ")
                llm_resAns = llm_resAns.strip()

            try:
                resConf = res["confidence"]
            except:
                resConf = res["probability"]
            confsQA.append(resConf)

            if llm_found:
                llm_resConf = llm_res["prob"] #llm_res["confidence"]

            gtAcc = []
            llm_gtAcc = []
            gtAnswers = [ans["answer"] for ans in gt["answers"]]

            if len(set(gtAnswers)) > 1:
                for ansDic in gt["answers"]:
                    ansDic["answer"] = self.processPunctuation(ansDic["answer"])
                    ansDic["answer"] = self.processDigitArticle(ansDic["answer"])
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)

                if llm_found:
                    llm_resAns = self.processPunctuation(llm_resAns)
                    llm_resAns = self.processDigitArticle(llm_resAns)
            
            #######################################################
            for gtAnsDatum in gt["answers"]:
                otherGTAns = [item for item in gt["answers"] if item != gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)

                if llm_found:
                    llm_matchingAns = [item for item in otherGTAns if item["answer"] == llm_resAns]
                    acc = min(1, float(len(llm_matchingAns)) / 3)
                    llm_gtAcc.append(acc)

            #######################################################
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            risk = 1.0 - avgGTAcc
            accQA.append(avgGTAcc)

            if llm_found:
                llm_avgGTAcc = float(sum(llm_gtAcc)) / len(llm_gtAcc)
                llm_risk = 1.0 - llm_avgGTAcc
                llm_accQA.append(llm_avgGTAcc)
            else:
                llm_avgGTAcc = None
                llm_risk = None
                llm_resConf = None
            
            llm_performance.append({'question_id': quesId, 'backbone_acc': avgGTAcc, 'llm_acc': llm_avgGTAcc, 'backbone_confidence': resConf, 'llm_confidence': llm_resConf})

            ########################################################
            self.setEvalQA(quesId, avgGTAcc, risk, resConf, llm_avgGTAcc, llm_risk, llm_resConf, is_threshold=is_threshold)
            ########################################################
        # dump llm performance to csv
        if dir is not None:
            pd.DataFrame(llm_performance).to_csv(os.path.join(dir,name+'llm_performance.csv'),index=False)
        self.plot_hist_per_acc(accQA, confsQA, dir, name)
        if not is_threshold:
            self.setAccuracy(accQA)
            self.setllmAccuracy(llm_accQA)
    
    def plot_hist_per_acc(self, accQA, confsQA, dir, name):
        if name is not None:
            name+='_'
        else:
            name=''
        f = plt.figure()
        hist,bin_edges=np.histogram(confsQA,bins=20)
        plt.bar(bin_edges[1:],hist/np.sum(hist),width=0.05,alpha=0.5,edgecolor='b')
        plt.xlabel('confidence')
        plt.ylabel('frequency')
        plt.title('Confidence distribution')
        plt.savefig(os.path.join(dir,name+'confidence_distribution.png'),bbox_inches='tight')
        f = plt.figure(figsize=(5,4))
        for a_i,a in enumerate([0.3,0.6,0.9,0,1]):
            idx = [i for i, x in enumerate(accQA) if np.abs(x-a) <0.01]
            hist,bin_edges=np.histogram([confsQA[i] for i in idx],bins=20)
            plt.plot(bin_edges[1:],hist,label='Acc:'+str(a),linewidth=2,alpha=0.8)
            plt.xlabel('confidence')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(os.path.join(dir,name+f'confidence_per_acc_{a}.png'),bbox_inches='tight')
        plt.yscale('log')
        plt.savefig(os.path.join(dir,name+f'confidence_per_acc_logscale.png'),bbox_inches='tight')
        f = plt.figure(figsize=(5,4))
        for a_i, a in enumerate([0.3, 0.6, 0.9, 0, 1]):
            idx = [i for i, x in enumerate(accQA) if np.abs(x - a) < 0.01]
            hist, bin_edges = np.histogram([confsQA[i] for i in idx], bins=20)
            plt.plot(bin_edges[1:], hist/sum(hist), label='Acc:' + str(a), linewidth=2, alpha=0.8)
            plt.xlabel('confidence')
            plt.ylabel('Frequency')
            plt.legend()
        plt.savefig(os.path.join(dir, name + f'confidence_per_acc_distribution.png'), bbox_inches='tight')
        return
    
    def processPunctuation(self, inText):
        outText = puncStripBegin.sub("", inText)
        outText = puncStripEnd.sub("", outText)
        outText = commaStrip.sub("", outText)
        outText = puncStrip.sub(" ", outText)
        outText = spaceCleanup.sub(" ", outText)
        outText = puncStrip2.sub(" ", outText)
        outText = puncStrip2.sub("", outText)
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = " ".join(outText)
        return outText

    def setAccuracy(self, accQA):
        self.accuracy["vqa_accuracy"] = round(
            100 * float(sum(accQA)) / len(accQA), self.n
        )
    def setllmAccuracy(self, accQA):
        if len(accQA)==0:
            self.accuracy["llm_vqa_accuracy"] = 0
        else:
            self.accuracy["llm_vqa_accuracy"] = round(
                100 * float(sum(accQA)) / len(accQA), self.n
            )
    def setllmDelegateAccuracy(self, accQA, delegate_threshold=0.5,delegate_percentage=None,ece=None):
        self.accuracy[f"delegate_accuracy_{delegate_threshold}"] = round(
            100 * float(sum(accQA)) / len(accQA), self.n
        )
        self.accuracy[f'delegate_percentage_{delegate_threshold}'] = round(delegate_percentage*100,self.n)
        if ece is not None:
            self.accuracy[f'delegate_ece_{delegate_threshold}'] = round(ece,self.n)

    def setEvalQA(self, quesId, acc, risk, conf, llm_acc, llm_risk, llm_conf, is_threshold=False):
        qa = self.evalThresholdQA if is_threshold else self.evalQA
        qa[quesId] = {
            "accuracy": round(100.0 * acc, self.n),
            "indiv_risk": risk,
            "confidence": float(conf),
            "llm_accuracy": round(100.0 * llm_acc, self.n) if llm_acc is not None else None,
            "llm_indiv_risk": llm_risk,
            "llm_confidence": float(llm_conf) if llm_conf is not None else None,
        }
