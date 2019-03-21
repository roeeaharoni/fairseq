# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn.functional as F
from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, models, tgt_dict):
        self.models = models
        self.pad = tgt_dict.pad()

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def score_batched_itr(self, data_itr, cuda=False, timer=None):
        """Iterate over a batched dataset and yield scored translations."""
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if timer is not None:
                timer.start()
            pos_scores, attn = self.score(s)
            for i, id in enumerate(s['id'].data):
                # remove padding from ref
                src = utils.strip_pad(s['net_input']['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                tgt_len = ref.numel()
                pos_scores_i = pos_scores[i][:tgt_len]
                score_i = pos_scores_i.sum() / tgt_len
                if attn is not None:
                    attn_i = attn[i]
                    _, alignment = attn_i.max(dim=0)
                else:
                    attn_i = alignment = None
                hypos = [{
                    'tokens': ref,
                    'score': score_i,
                    'attention': attn_i,
                    'alignment': alignment,
                    'positional_scores': pos_scores_i,
                }]
                if timer is not None:
                    timer.stop(s['ntokens'])
                # return results in the same format as SequenceGenerator
                yield id, src, ref, hypos

    def score(self, sample):
        """Score a batch of translations."""
        net_input = sample['net_input']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in self.models:
            with torch.no_grad():
                model.eval()
                decoder_out = model.forward(**net_input)
                attn = decoder_out[1]

            probs = model.get_normalized_probs(decoder_out, log_probs=False, sample=sample).data
            print(probs.shape)
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs.div_(len(self.models))

        # ours:
        ens_selected_probs_per_step = torch.tensor(avg_probs.gather(
            dim=2,
            index=sample['target'].data.unsqueeze(-1),
        )).squeeze()
        # end ours

        avg_probs.log_()

        # ours:
        flattend_ens_log_probs = avg_probs.squeeze()
        ens_ents_per_step = self.entropy(flattend_ens_log_probs)
        # end ours

        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        avg_probs = avg_probs.gather(
            dim=2,
            index=sample['target'].data.unsqueeze(-1),
        )

        # ours:
        # per step
        # get ens_globally_selected_prob (mean) - done
        # ens std
        #  top_k_ens_, entropy
        # print(avg_probs.shape) # [1,seq_len,1]
        print("probs: ", ens_selected_probs_per_step)
        print("ents: ", ens_ents_per_step)

        # path = "/home/nlp/aharonr6/git/nmt-uncertainty/models/en_he_trans_base_seg_ens/force_decode_features.txt"
        path = "/home/nlp/aharonr6/git/nmt-uncertainty/data/QE/WMT17 Quality Estimation Shared Task Training and Development Data/sentence_level/preprocessed/train_force_decode_features.txt"
        with open(path, "a") as file:
            ens_probs_floats = [str(float(x.item())) for x in ens_selected_probs_per_step]
            ent_ents_floats = [str(float(x.item())) for x in ens_ents_per_step]
            file.write(" ".join(ens_probs_floats) + "\t" + " ".join(ent_ents_floats) + "\n")
        # end ours

        return avg_probs.squeeze(2), avg_attn

    def entropy(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean(dim=1)
        return b