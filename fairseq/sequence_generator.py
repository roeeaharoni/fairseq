# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
from collections import defaultdict
import pickle

import torch.nn.functional as F

from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder

import sys
import numpy as np


class SequenceGenerator(object):
    def __init__(
            self, models, tgt_dict, beam_size=1, minlen=1, maxlen=None, stop_early=True,
            normalize_scores=True, len_penalty=1, unk_penalty=0, retain_dropout=False,
            sampling=False, sampling_topk=-1, sampling_temperature=1,
            diverse_beam_groups=-1, diverse_beam_strength=0.5,
    ):
        """Generates translations of a given source sentence.
        Args:
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        """
        self.agreement_structs = []
        self.top_k_words = 10
        # self.top_k_entropy = 100
        self.agreement_batch_struct = {}  # defaultdict(lambda: [])
        self.models = models
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        max_decoder_len = min(m.max_decoder_positions() for m in self.models)
        max_decoder_len -= 1  # we define maxlen not including the EOS marker
        self.maxlen = max_decoder_len if maxlen is None else min(maxlen, max_decoder_len)
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout

        ### A&R
        self.tgt_dict = tgt_dict
        ###

        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_temperature)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        else:
            self.search = search.BeamSearch(tgt_dict)

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def generate_batched_itr(
            self, data_itr, beam_size=None, maxlen_a=0.0, maxlen_b=None,
            cuda=False, timer=None, prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """
        if maxlen_b is None:
            maxlen_b = self.maxlen

        ###
        batch_count = 0
        ###
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if 'net_input' not in s:
                continue
            input = s['net_input']
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items()
                if k != 'prev_output_tokens'
            }
            srclen = encoder_input['src_tokens'].size(1)
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    encoder_input,
                    beam_size=beam_size,
                    maxlen=int(maxlen_a * srclen + maxlen_b),
                    prefix_tokens=s['target'][:, :prefix_size] if prefix_size > 0 else None,
                )
                ### R&A
                # self.log_to_analysis_file(batch_count, encoder_input, hypos)
                batch_count += 1
                ###

            if timer is not None:
                timer.stop(sum(len(h[0]['tokens']) for h in hypos))
            for i, id in enumerate(s['id'].data):
                # remove padding
                src = utils.strip_pad(input['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                yield id, src, ref, hypos[i]

    def log_to_analysis_file(self, batch_count, encoder_input, hypos):
        final_batch_result = {"agreements_over_time": self.agreement_batch_struct,
                              "final_hypos": hypos}
        final_batch_result["source"] = encoder_input['src_tokens']
        self.agreement_structs.append(final_batch_result)
        self.agreement_batch_struct = {}

        self.write_analysis_to_file(batch_count)

    def write_analysis_to_file(self, batch_count):
        BATCH_SIZE = 1
        slim = True
        # NUM_EXAMPLES = 4515
        # PICKLE_BATCHES = NUM_EXAMPLES // BATCH_SIZE - 1
        # if batch_count > PICKLE_BATCHES:
        if batch_count == 4000:
            if not slim:
                final_eval_result = self.final_result(self.agreement_structs, slim=slim)
                fname = "ens_eval"
            else:
                final_eval_result = self.final_result(self.agreement_structs, slim=slim)
                fname = "ens_eval_slim"

            with open("/home/nlp/aharonr6/git/nmt-uncertainty/models/en_he_trans_base_seg_ens/{}_b{}_k{}_global_baseline.pkl".format(
                    fname, batch_count*BATCH_SIZE, self.top_k_words), "wb") as f:
                pickle.dump(final_eval_result, f, pickle.HIGHEST_PROTOCOL)
            # exit()

    def extract_prefix_to_entropies_and_probabilities(self, agreements_over_time, source_batch):
        """

        {"tokens": tokens, "strings": self.tgt_dict.string(tokens).split("\n"),
                "model_probs": model_probs,
                    "ens_prob": ensemble_prob,
                    "agreements": self._calc_agreement(model_probs, ensemble_prob)}
        {"ens": ens_agreement, "models": models_agreement}


        :param agreements_over_time:
        :return:
        """
        mapping_ens_ent = {}  # mapping from prefix to ensemble entropy
        mapping_models_ent = {}  # mapping from prefix to model entropies
        mapping_ens_prob = {}
        mapping_models_prob = {}

        mapping_top_k_models_probs = {}
        mapping_top_k_ens_prob = {}
        mapping_argtop_k_models_probs = {}
        mapping_argtop_k_ens_prob = {}
        mapping_top_k_models_ents = {}
        mapping_top_k_ens_ent = {}
        mapping_ens_top_k_models_probs = {}

        for step in agreements_over_time:
            step_info = agreements_over_time[step]
            # print("STEP:\n", step)
            # print("TOKENS_PER_PREFIX_PER_HYPO_PER_SAMPLE:\n", step_info["tokens"], len(step_info["tokens"]))
            # print("should be of size step*(batch_size*beam_size)")
            # print("SRC_TOKENS_PER_SAMPLE", step_info["source_tokens"], len(step_info["source_tokens"]))
            # print("should be of size 16*seq_len")

            source_strings_in_batch = self.tgt_dict.string(source_batch).split("\n")
            # print("source_strings_in_batch:\n", source_strings_in_batch)

            tokens_per_prefix = self.tgt_dict.string(torch.tensor(step_info["tokens"])).split("\n")
            # print("step_prefixes_strings:\n", tokens_per_prefix)

            for ix, prefix in enumerate(tokens_per_prefix):
                # for each output prefix and source sequence, create key
                # print("prefix {} prints:".format(ix))

                # TODO: this works only for batch_size=1
                key = (prefix, source_strings_in_batch[0])

                mapping_ens_prob[key] = step_info["ens_prob"][ix]
                mapping_models_prob[key] = [model_[ix] for model_ in step_info["model_probs"]]
                mapping_ens_ent[key] = step_info["agreements"]["ens"][ix]
                mapping_models_ent[key] = [model_[ix] for model_ in step_info["agreements"]["models"]]

                # self.top_k_words = k
                mapping_top_k_models_probs[key] = [prob.topk(self.top_k_words) for prob in mapping_models_prob[key]]
                mapping_argtop_k_models_probs[key] = [prob[1] for prob in mapping_top_k_models_probs[key]]
                mapping_top_k_models_probs[key] = [prob[0] for prob in mapping_top_k_models_probs[key]]
                mapping_top_k_ens_prob[key] = mapping_ens_prob[key].topk(self.top_k_words)
                mapping_argtop_k_ens_prob[key] = mapping_top_k_ens_prob[key][1]
                mapping_top_k_ens_prob[key] = mapping_top_k_ens_prob[key][0]

                mapping_ens_top_k_models_probs[key] = [prob[mapping_argtop_k_ens_prob[key]] for prob in mapping_models_prob[key]]

                # .view() as it expects a batch
                mapping_top_k_ens_ent[key] = self.entropy(mapping_top_k_ens_prob[key].view(1, self.top_k_words))
                mapping_top_k_models_ents[key] = [self.entropy(prob.view(1, self.top_k_words)) for prob in \
                                                  mapping_top_k_models_probs[key]]


        return mapping_models_prob, mapping_ens_prob, mapping_ens_ent, mapping_models_ent, \
               mapping_top_k_models_probs, mapping_top_k_ens_prob, mapping_top_k_models_ents, mapping_top_k_ens_ent, \
               mapping_argtop_k_models_probs, mapping_argtop_k_ens_prob, mapping_ens_top_k_models_probs

    def eprint(self, *args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    def final_result(self, agreement_structs, slim=True):
        # list of instances
        # only top beam
        # list of steps (over time)
        # * model entropies of next token
        # * ensemble entropy of next token
        # * selected next token per model
        # * selected next token of ensemble
        # * prefix score

        samples = []
        for batch_ix, batch in enumerate(agreement_structs):
            prefix_to_models_probs, \
            prefix_to_ens_prob, \
            prefix_to_ens_entropies, \
            prefix_to_models_entropies, \
            prefix_to_models_top_k_probs, \
            prefix_to_ens_top_k_prob, \
            prefix_to_models_top_k_ents, \
            prefix_to_ens_top_k_ent, \
            prefix_to_argtop_k_models_probs, \
            prefix_to_argtop_k_ens_prob, \
            prefix_to_ens_top_k_models_probs \
                = self.extract_prefix_to_entropies_and_probabilities(
                batch["agreements_over_time"], batch["source"])

            for sample_ix, sample in enumerate(batch["final_hypos"]):
                self.eprint("Processing: ", len(samples))

                hypos = []

                source_tokens = batch["source"][sample_ix]
                source_info = {"source_tokens": source_tokens.cpu().numpy(),
                               "source_str": self.tgt_dict.string(source_tokens)}

                for hypo in sample[:1 if slim else len(sample)]:
                    info = {}

                    info["target"] = hypo["tokens"]
                    info["target_str"] = self.tgt_dict.string(hypo["tokens"])

                    info_over_time = []
                    for i in range(len(info["target"])):
                        if slim:
                            self.generate_step_info_slim(hypo, i, info, info_over_time, prefix_to_ens_entropies,
                                                         prefix_to_models_entropies,
                                                         prefix_to_models_top_k_probs, prefix_to_ens_top_k_prob,
                                                         prefix_to_models_top_k_ents, prefix_to_ens_top_k_ent,
                                                         prefix_to_argtop_k_models_probs, prefix_to_argtop_k_ens_prob,
                                                         prefix_to_ens_top_k_models_probs,
                                                         source_info,
                                                         prefix_to_models_probs,
                                                         prefix_to_ens_prob)
                        else:
                            self.generate_step_info(hypo, i, info, info_over_time, prefix_to_ens_entropies,
                                                    prefix_to_ens_prob,
                                                    prefix_to_models_entropies, prefix_to_models_probs, source_info)

                    info["target"] = info["target"].cpu().numpy()
                    info["per_token"] = info_over_time
                    hypos.append(info)

                samples.append({"targets": hypos, "source": source_info})
                # samples.append(info)

        return samples

    def generate_step_info_slim(self, hypo, i, info, info_over_time, prefix_to_ens_entropies,
                                prefix_to_models_entropies, prefix_to_models_top_k_probs,
                                prefix_to_ens_top_k_prob, prefix_to_models_top_k_ents, prefix_to_ens_top_k_ent,
                                prefix_to_argtop_k_models_probs, prefix_to_argtop_k_ens_prob,
                                prefix_to_ens_top_k_models_probs,
                                source_info,
                                prefix_to_models_probs,
                                prefix_to_ens_prob):
        prefix = info["target"][:i]
        prefix = self.tgt_dict.string(prefix)
        key = (prefix, source_info["source_str"])
        step_info = {"prefix": prefix,
                     "models_ents": [v.cpu().numpy() for v in prefix_to_models_entropies[key]],
                     "ens_ent": prefix_to_ens_entropies[key].cpu().numpy(),
                     "step_score": hypo["positional_scores"][i].cpu().numpy(),
                     "models_top_k_probs": [v.cpu().numpy() for v in prefix_to_models_top_k_probs[key]],
                     "models_top_k_ents": [v.cpu().numpy() for v in prefix_to_models_top_k_ents[key]],
                     "ens_top_k_prob": prefix_to_ens_top_k_prob[key].cpu().numpy(),
                     "ens_top_k_ent": prefix_to_ens_top_k_ent[key].cpu().numpy(),
                     "ens_argtop_k": prefix_to_argtop_k_ens_prob[key].cpu().numpy(),
                     "models_argtop_k": [v.cpu().numpy() for v in prefix_to_argtop_k_models_probs[key]],
                     "ens_topk_models_probs": [v.cpu().numpy() for v in prefix_to_ens_top_k_models_probs[key]]
                     }
        step_info["selected_token_per_model"] = [argtop[0].cpu().numpy() for argtop in
                                                 prefix_to_argtop_k_models_probs[key]]
        step_info["selected_token_by_ens"] = prefix_to_argtop_k_ens_prob[key][0].cpu().numpy()

        step_info["selected_token_per_model_str"] = [self.tgt_dict.string(v[0].view((1, 1))) for v in
                                                     prefix_to_argtop_k_models_probs[key]]
        step_info["selected_token_by_ens_str"] = self.tgt_dict.string(prefix_to_argtop_k_ens_prob[key][0].view((1, 1)))

        step_info["ens_argtop_k_str"] = self.tgt_dict.string(prefix_to_argtop_k_ens_prob[key].view((self.top_k_words, 1)))
        step_info["models_argtop_k_str"] = [self.tgt_dict.string(v.view((self.top_k_words, 1))) for v in
                                            prefix_to_argtop_k_models_probs[key]]

        step_info["globally_selected_token"] = info["target"][i].cpu().numpy()
        step_info["globally_selected_token_ens_prob"] = prefix_to_ens_prob[key][step_info["globally_selected_token"]].cpu().numpy()
        step_info["globally_selected_token_models_probs"] = [probs[step_info["globally_selected_token"]].cpu().numpy()
                                                             for probs in prefix_to_models_probs[key]]
        # self.eprint(step_info)


        info_over_time.append(step_info)

    def generate_step_info(self, hypo, i, info, info_over_time, prefix_to_ens_entropies, prefix_to_ens_prob,
                           prefix_to_models_entropies, prefix_to_models_probs, source_info):
        """DEPRECATED"""
        prefix = info["target"][:i]
        prefix = self.tgt_dict.string(prefix)
        key = (prefix, source_info["source_str"])
        step_info = {"prefix": prefix,
                     "models_probs": prefix_to_models_probs[key],
                     "models_ents": [v.cpu().numpy() for v in prefix_to_models_entropies[key]],
                     "ens_ent": prefix_to_ens_entropies[key].cpu().numpy(),
                     "ens_prob": prefix_to_ens_prob[key],
                     "step_score": hypo["positional_scores"][i].cpu().numpy()}
        step_info["selected_token_per_model"] = [torch.max(model_prob, 0)[1] for model_prob in
                                                 prefix_to_models_probs[key]]
        step_info["selected_token_by_ens"] = torch.max(prefix_to_ens_prob[key], 0)[1]
        step_info["selected_token_per_model_str"] = [self.tgt_dict.string(v.view((1, 1))) for v in
                                                     step_info["selected_token_per_model"]]
        step_info["selected_token_by_ens_str"] = self.tgt_dict.string(step_info["selected_token_by_ens"].view((1, 1)))
        info_over_time.append(step_info)

    def final_result_(self, agreement_structs):
        """
        DEPRECATED

        :param agreement_structs:
        :return:
        """
        # list of instances
        # list of beams
        # list of steps (over time)
        # * model probabilities of next token
        # * ensemble probability of next token
        # * model entropies of next token
        # * ensemble entropy of next token
        # * selected next token per model
        # * selected next token of ensemble
        # * prefix score

        samples = []
        for batch_ix, batch in enumerate(agreement_structs):
            prefix_to_models_probs, \
            prefix_to_ens_prob, \
            prefix_to_ens_entropies, \
            prefix_to_models_entropies, \
                = self.extract_prefix_to_entropies_and_probabilities(
                batch["agreements_over_time"], batch["source"])

            for sample_ix, sample in enumerate(batch["final_hypos"]):
                hypos = []

                # TODO: use src dict. we will use target dict to convert to string
                # TODO: works for now because we use joint bpe
                source_tokens = batch["source"][sample_ix]
                source_info = {"source_tokens": source_tokens.cpu().numpy(),
                               "source_str": self.tgt_dict.string(source_tokens)}

                for hypo in sample:
                    info = {}

                    info["target"] = hypo["tokens"]
                    info["target_str"] = self.tgt_dict.string(hypo["tokens"])

                    info_over_time = []

                    for i in range(len(info["target"])):
                        # step_info = {}

                        prefix = info["target"][:i]
                        prefix = self.tgt_dict.string(prefix)

                        key = (prefix, source_info["source_str"])

                        # print(list(prefix_to_models_probs), "target:", info["target"], "prefix:", prefix, i, "target str", self.tgt_dict.string(info["target"]))

                        step_info = {"prefix": prefix,
                                     "models_probs": prefix_to_models_probs[key],
                                     "models_ents": [v.cpu().numpy() for v in prefix_to_models_entropies[key]],
                                     "ens_prob": prefix_to_ens_prob[key],
                                     "ens_ent": prefix_to_ens_entropies[key].cpu().numpy(),
                                     "step_score": hypo["positional_scores"][i].cpu().numpy()}

                        step_info["selected_token_per_model"] = [torch.max(model_prob, 0)[1] for model_prob in
                                                                 step_info["models_probs"]]
                        step_info["selected_token_by_ens"] = torch.max(step_info["ens_prob"], 0)[1]

                        # print(torch.max(step_info[i]["ens_prob"], 0)[1])

                        step_info["selected_token_per_model_str"] = [self.tgt_dict.string(v.view((1, 1))) for v in
                                                                     step_info["selected_token_per_model"]]
                        step_info["selected_token_by_ens_str"] = self.tgt_dict.string(
                            step_info["selected_token_by_ens"].view((1, 1)))

                        step_info["models_probs"] = [v.cpu().numpy() for v in step_info["models_probs"]]
                        step_info["ens_prob"] = step_info["ens_prob"].cpu().numpy()

                        info_over_time.append(step_info)

                    info["target"] = info["target"].cpu().numpy()
                    info["per_token"] = info_over_time
                    hypos.append(info)

                samples.append({"targets": hypos, "source": source_info})

        return samples

    def generate(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None):
        """Generate a batch of translations.

        Args:
            encoder_input: dictionary containing the inputs to
                model.encoder.forward
            beam_size: int overriding the beam size. defaults to
                self.beam_size
            max_len: maximum length of the generated sequence
            prefix_tokens: force decoder to begin with these tokens
        """
        with torch.no_grad():
            return self._generate(encoder_input, beam_size, maxlen, prefix_tokens)

    def _generate(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None):
        """See generate"""
        src_tokens = encoder_input['src_tokens']
        bsz, srclen = src_tokens.size()
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = beam_size if beam_size is not None else self.beam_size
        beam_size = min(beam_size, self.vocab_size - 1)

        encoder_outs = []
        incremental_states = {}
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            if isinstance(model.decoder, FairseqIncrementalDecoder):
                incremental_states[model] = {}
            else:
                incremental_states[model] = None

            # compute the encoder output for each beam
            encoder_out = model.encoder(**encoder_input)
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
            new_order = new_order.to(src_tokens.device)
            encoder_out = model.encoder.reorder_encoder_out(encoder_out, new_order)
            encoder_outs.append(encoder_out)

        # initialize buffers
        scores = src_tokens.data.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos
        attn, attn_buf = None, None
        nonpad_idxs = None

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.
            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.
            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                for i, model in enumerate(self.models):
                    if isinstance(model.decoder, FairseqIncrementalDecoder):
                        model.decoder.reorder_incremental_state(incremental_states[model], reorder_state)
                    encoder_outs[i] = model.encoder.reorder_encoder_out(encoder_outs[i], reorder_state)

            lprobs, avg_attn_scores = self._decode(tokens[:, :step + 1], encoder_outs, incremental_states,
                                                   encoder_input["src_tokens"])

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), maxlen + 2)
                    attn_buf = attn.clone()
                    nonpad_idxs = src_tokens.ne(self.pad)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    probs_slice = lprobs.view(bsz, -1, lprobs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=prefix_tokens[:, step].view(-1, 1).data
                    ).expand(-1, cand_size)
                    cand_indices = prefix_tokens[:, step].view(-1, 1).expand(bsz, cand_size).data
                    cand_beams = torch.zeros_like(cand_indices)
                else:
                    cand_scores, cand_indices, cand_beams = self.search.step(
                        step,
                        lprobs.view(bsz, -1, self.vocab_size),
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                    )
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(
                    step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            finalized_sents = set()
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(
                        step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)

                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized

    def _decode(self, tokens, encoder_outs, incremental_states, encoder_input_tokens):
        if len(self.models) == 1:
            return self._decode_one(tokens, self.models[0], encoder_outs[0], incremental_states, log_probs=True)

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(tokens, model, encoder_out, incremental_states, log_probs=True)
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        log_probs_stacked = torch.stack(log_probs, dim=0)
        avg_probs = torch.logsumexp(log_probs_stacked, dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))

        ##### new score
        # std = torch.std(log_probs_stacked, dim=0) # (v, b)
        # avg_probs = torch.logsumexp(torch.stack([-std, avg_probs], dim=0), dim=0)
        # del std
        #####
        # print(encoder_outs)
        # print(encoder_outs[0]["encounter_outs"].size())
        # exit()
        self.calc_and_save_agreement(tokens, log_probs, avg_probs, encoder_outs)
        #####

        return avg_probs, avg_attn

    def _calc_agreement(self, model_probs, ensemble_prob):
        ens_agreement = self.entropy(ensemble_prob).cpu()
        models_agreement = [self.entropy(model_prob).cpu() for model_prob in model_probs]
        return {"ens": ens_agreement, "models": models_agreement}

    def calc_and_save_agreement(self, tokens, model_probs, ensemble_prob, source_tokens):
        """
        :param tokens:
        :param model_probs: a list of model probabilities for each model from the ensemble
        :param ensemble_prob: the ensemble distribution (e.g. avg)
        :return: None
        """

        # print("1\n", model_probs, "\n2\n", ensemble_prob, "\n3\n", tokens)

        self.agreement_batch_struct[len(tokens[0])] = {"source_tokens": source_tokens,
                                                       "tokens": tokens.cpu(),
                                                       "strings": self.tgt_dict.string(tokens).split("\n"),
                                                       "model_probs": [model_prob.cpu() for model_prob in model_probs],
                                                       "ens_prob": ensemble_prob.cpu(),
                                                       "agreements": self._calc_agreement(model_probs, ensemble_prob)}

        # print(self.agreement_batch_struct[len(tokens[0])]["tokens"], self.agreement_batch_struct[len(tokens[0])]["strings"])
        # print(self.agreement_struct)

    def _decode_one(self, tokens, model, encoder_out, incremental_states, log_probs):
        with torch.no_grad():
            if incremental_states[model] is not None:
                decoder_out = list(model.decoder(tokens, encoder_out, incremental_state=incremental_states[model]))
            else:
                decoder_out = list(model.decoder(tokens, encoder_out))
            decoder_out[0] = decoder_out[0][:, -1, :]
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn['attn']
            if attn is not None:
                if type(attn) is dict:
                    attn = attn['attn']
                attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        return probs, attn

    def entropy(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean(dim=1)
        return b
