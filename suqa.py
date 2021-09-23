import os
import re
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import logging

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
)


class SuQA:
    def __init__(self, device, config):
        self.suqa_model = XQA(device, config)

    @staticmethod
    def from_pretrained(device, path_qam, path_explainer):
        config = {
            "qa_model": "allenai/unifiedqa-t5-base",
            "generator": "sshleifer/distilbart-cnn-12-6",
            "decode_num_beams": 1,
            "load_qamodel": path_qam,
            "gen_input_max_len": 512,
            "gen_max_len": 512,
        }
        config = SimpleNamespace(**config)

        suqa = SuQA(device, config)
        suqa.suqa_model.load(path_explainer)
        suqa.suqa_model.load_qa_model()
        return suqa

    def predict(self, batch):
        """
        batch: [(question1, passage1), (question2, passage2), ...]
        return: [(explanation1, answer1), (explanation2, answer2), ...]
        """
        with torch.no_grad():
            questions, contexts = zip(*batch)
            _, _, outputs = self.suqa_model.generate_expl(questions, contexts)
            sample_expls = self.suqa_model.gen_tok.batch_decode(outputs.sequences, skip_special_tokens=True)
            sample_expls = list(map(str.strip, sample_expls))
            answers = self.suqa_model.predict_answer(sample_expls, questions)

        return list(zip(sample_expls, answers["predicted_answers"]))


class QAModel:
    def __init__(self, device, args):
        self.device = device
        self.args = args

        self.qa_tok = AutoTokenizer.from_pretrained(args.qa_model)

        self.answer_max_len = 256
        self.input_max_len = 512

        if args.qa_model.startswith("allenai/unifiedqa-t5"):
            self.qa_model_type = "gen"
            self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(args.qa_model,
                                                                  return_dict_in_generate=True,
                                                                  ).to(device)

        else:
            self.qa_model_type = "span"
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model).to(device)

        self.qa_model.eval()

    def predict_batch(self, batch, return_loss=False):
        inputs = self.prepare_input(batch).to(self.device)
        out_dict = {}

        if self.qa_model_type == "gen":
            outputs = self.qa_model.generate(input_ids=inputs.input_ids,
                                             attention_mask=inputs.attention_mask,
                                             num_beams=1, num_beam_groups=1, do_sample=False,
                                             output_scores=True,
                                             max_length=self.qa_model.config.max_length)
            predicted_answers = self.qa_tok.batch_decode(outputs.sequences,
                                                         skip_special_tokens=True)
            out_dict["score"] = get_gen_scores(self.qa_model, outputs)

            if return_loss:
                labels = self.qa_tok(list(map(str, batch["answer"])), padding=True, truncation=True,
                                     max_length=self.answer_max_len)
                label_len = len(labels[0])

                # Pad token to -100
                labels = [[tk if tk != self.qa_tok.pad_token_id else -100 for tk in ans] for ans in labels.input_ids]

                # If there is no gold answer span in the paragraph, ignore them.
                labels = [[-100] * label_len if ans.lower() not in para.lower() else lbl
                          for lbl, ans, para in zip(labels, batch["answer"], batch["context"])]

                labels = torch.tensor(labels, device=self.device)

                self.qa_model.train()

                output = self.qa_model(input_ids=inputs.input_ids,
                                       attention_mask=inputs.attention_mask,
                                       labels=labels,
                                       )
                out_dict["loss"] = output.loss

                self.qa_model.eval()

        else:
            with torch.no_grad() if not return_loss else nullcontext():
                outputs = self.qa_model(input_ids=inputs.input_ids,
                                        attention_mask=inputs.attention_mask,
                                        )

            best_scores = list(zip(outputs.start_logits.max(1)[0].cpu().tolist(), outputs.end_logits.max(1)[0].cpu().tolist()))
            best_spans = list(zip(outputs.start_logits.argmax(1).cpu().tolist(), outputs.end_logits.argmax(1).cpu().tolist()))
            answer_sequences = [input_ids[start_span:end_span + 1]
                                for input_ids, (start_span, end_span) in zip(inputs.input_ids, best_spans)]
            predicted_answers = self.qa_tok.batch_decode(answer_sequences,
                                                         skip_special_tokens=True)
            out_dict["span"] = best_spans
            out_dict["score"] = best_scores

            if return_loss:
                assert False, "Not implemented."

        out_dict.update({
            "predicted_answers": predicted_answers,
            "qa_outputs": outputs
        })

        out_dict["predicted_answers"] = list(map(lambda x: x.replace("no answer>", ""), out_dict["predicted_answers"]))

        return out_dict

    def prepare_input(self, batch):
        normalize_text = lambda x: re.sub("'(.*)'", r"\1", x.lower())

        qa_inputs_text = list()

        for p, q in zip(batch["context"], batch["question"]):
            if self.args.qa_model.startswith("allenai/unifiedqa-t5"):
                qa_inputs_text.append(normalize_text(f"{q} \\n {p}"))
            else:
                qa_inputs_text.append(f"{q} {self.qa_tok.sep_token} yes no {self.qa_tok.sep_token} {p}")

        return self.qa_tok.batch_encode_plus(qa_inputs_text,
                                             padding=True, add_special_tokens=True,
                                             max_length=self.input_max_len, truncation=True,
                                             return_tensors="pt",
                                             )


class XQA(torch.nn.Module):
    def __init__(self, device, args):
        super().__init__()

        self.args = args
        self.device = device

        # Generator
        logging.info(f"Loading summarizer {args.generator}...")
        self.gen_tok = AutoTokenizer.from_pretrained(args.generator)

        if "distilbart" in args.generator:
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.generator,
                                                                   return_dict_in_generate=True,
                                                                   force_bos_token_to_be_generated=False).to(device)
            self.gen_model.config.forced_bos_token_id = None
        else:
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.generator,
                                                                   return_dict_in_generate=True,
                                                                   ).to(device)
            self.gen_model.config.forced_bos_token_id = None

        self.training_mode = False

        self.gen_model.eval()
        self.gen_num_beams = self.args.decode_num_beams
        self.gen_diversity_penalty = 0.1

        # Classifier
        logging.info(f"Loading downstream processor {args.qa_model}...")
        self.qa = QAModel(device, args)

    def load_qa_model(self):
        if self.args.load_qamodel is not None:
            fn_qa_model = self.args.load_qamodel
            logging.info(f"Loading {fn_qa_model}...")
            self.qa.qa_model.load_state_dict(torch.load(fn_qa_model))

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        if not os.path.exists(fn):
            logging.info(f"{fn} does not exist. Not loaded.")
            return

        logging.info(f"Loading {fn}...")
        ret = self.load_state_dict(torch.load(fn), strict=False)

        if len(ret.missing_keys) > 0:
            logging.info(f"Missing keys: {ret.missing_keys}")

        if len(ret.unexpected_keys) > 0:
            logging.info(f"Unexpected keys: {ret.unexpected_keys}")

    @contextmanager
    def eval_mode(self):
        restore_gen_mode = self.training_mode
        self.gen_model.eval()
        self.training_mode = False

        yield

        if restore_gen_mode:
            self.train_mode()

    def create_gen_input(self, q, t):
        if self.args.generator.startswith("t5"):
            return f"summarize: {q.strip()} {t.strip()}"

        return f"{q.strip()} {t.strip()}"

    def generate_expl(self, questions, contexts, temp=0, seed=1985, num_return_sequences=1,
                      encoder_outputs=None, attention_mask=None):
        mode = "greedy"

        if temp > 0:
            mode = "sampling"
        elif temp == 0 and self.gen_num_beams > 1:
            mode = "beam"

        gen_inputs = self.gen_tok([self.create_gen_input(q, t) for q, t in zip(questions, contexts)],
                                  padding=True,
                                  truncation=True,
                                  max_length=self.args.gen_input_max_len,
                                  return_tensors="pt")

        gen_input_ids = gen_inputs.input_ids.to(self.device)

        if attention_mask is None and encoder_outputs is None:
            attention_mask = gen_inputs.attention_mask.to(self.device)
            encoder_outputs = self.gen_model.get_encoder()(
                input_ids=gen_input_ids, attention_mask=attention_mask,
                output_hidden_states=True, output_attentions=True,
            )

        logits_processor = self.gen_model._get_logits_processor(
            repetition_penalty=None,
            no_repeat_ngram_size=0,
            encoder_no_repeat_ngram_size=0,
            encoder_input_ids=gen_input_ids,
            bad_words_ids=None,
            min_length=5,
            max_length=self.args.gen_max_len,
            eos_token_id=self.gen_model.config.eos_token_id,
            prefix_allowed_tokens_fn=None,
            num_beams=1,
            num_beam_groups=1,
            diversity_penalty=None,
            forced_bos_token_id=self.gen_model.config.forced_bos_token_id,
            forced_eos_token_id=self.gen_model.config.forced_eos_token_id,
        )

        stopping_criteria = self.gen_model._get_stopping_criteria(
            max_length=self.args.gen_max_len,
            max_time=None,
        )

        dec_input_ids = self.gen_model._prepare_decoder_input_ids_for_generation(gen_input_ids)

        if mode == "sampling":
            torch.manual_seed(seed)

            logits_warper = self.gen_model._get_logits_warper(
                top_k=None, top_p=None, temperature=temp, num_beams=1
            )

            if num_return_sequences == 1:
                with torch.no_grad():
                    outputs = self.gen_model.sample(dec_input_ids,
                                                    attention_mask=attention_mask,
                                                    encoder_outputs=encoder_outputs,
                                                    max_length=self.args.gen_max_len,
                                                    use_cache=True,
                                                    pad_token_id=self.gen_model.config.pad_token_id,
                                                    eos_token_id=self.gen_model.config.eos_token_id,
                                                    logits_processor=logits_processor,
                                                    logits_warper=logits_warper,
                                                    stopping_criteria=stopping_criteria,
                                                    )

            else:
                outputs = self.gen_model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=temp,
                    top_k=None,
                    top_p=None,
                    return_dict_in_generate=True,
                    min_length=5,
                    max_length=self.args.gen_max_len,
                    num_beams=1,
                    num_beam_groups=1,
                    num_return_sequences=num_return_sequences,
                )

        elif mode == "greedy":
            with torch.no_grad():
                outputs = self.gen_model.greedy_search(dec_input_ids,
                                                       attention_mask=attention_mask,
                                                       encoder_outputs=encoder_outputs,
                                                       max_length=self.args.gen_max_len,
                                                       use_cache=True,
                                                       pad_token_id=self.gen_model.config.pad_token_id,
                                                       eos_token_id=self.gen_model.config.eos_token_id,
                                                       logits_processor=logits_processor,
                                                       stopping_criteria=stopping_criteria,
                                                       output_scores=True,
                                                       )

        elif mode == "beam":
            outputs = self.gen_model.generate(
                input_ids=gen_input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                min_length=5,
                max_length=self.args.gen_max_len,
                num_beams=self.gen_num_beams,
                num_beam_groups=self.gen_num_beams,
                num_return_sequences=self.gen_num_beams,
                early_stopping=True,
                diversity_penalty=self.gen_diversity_penalty,
            )

        return encoder_outputs, attention_mask, outputs

    def predict_answer(self, sample_expls, questions, answers=None, return_loss=False):
        if answers is None:
            answers = [None] * len(questions)

        # Run QA model to predict answers from each explanation.
        qa_inputs_text = list()

        for q, p, ans in zip(questions, sample_expls, answers):
            qa_inputs_text.append({
                "context": p,
                "question": q,
                "answer": ans
            })

        qa_inputs_text = {k: [inst[k] for inst in qa_inputs_text] for k in qa_inputs_text[0].keys()}

        outputs = self.qa.predict_batch(qa_inputs_text, return_loss=return_loss)

        out_dict = {
            "predicted_answers": outputs["predicted_answers"],
            "score": outputs["score"],
            "loss": None,
        }

        if return_loss:
            out_dict["loss"] = outputs["loss"]

        return out_dict


def get_gen_scores(model, outputs):
    seqs = outputs.sequences.cpu().numpy()
    outputs_scores_lsm = list()
    final_scores = list()

    for j in range(len(outputs.scores)):
        outputs_scores_lsm.append(outputs.scores[j].log_softmax(-1))

    for i, seq in enumerate(seqs[:, 1:]):
        scores = list()

        for j, tk_id in enumerate(seq):
            if tk_id == model.config.pad_token_id:
                break

            scores.append(outputs_scores_lsm[j][i][tk_id].item())

        final_scores.append(scores)

    return final_scores
