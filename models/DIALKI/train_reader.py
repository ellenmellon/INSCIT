import collections
import json
import os
from typing import List
import time
import heapq

import argparse
import glob
import logging
import math
import numpy as np
import torch
import transformers as tfs

import config
from data_utils import data_collator, reader_dataset
from data_utils import utils as du
from data_utils import data_class
import eval
import models
from models import loss
from utils import checkpoint
from utils import dist_utils
from utils import model_utils
from utils import options
from utils import sampler
from utils import utils

try:
    from apex import amp
except:
    pass

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

ReaderQuestionPredictions = collections.namedtuple(
    'ReaderQuestionPredictions',
    ['id', 'gold_answers', 'passage_spans', 'passage_answers'])


class ReaderTrainer(object):
    
    def __init__(self, args):

        utils.print_section_bar('Initializing components for training')
        
        self.topk_spans = 1
        self.topk_passages = 10
        self.topk_cp_info_filename = 'topk_cp_info.json'

        checkpoint_file = None
        if args.checkpoint_file:
            checkpoint_file = args.checkpoint_file
        else:
            # eval mode: get the best model automatically
            if args.train_file is None:
                checkpoint_file = os.path.join(
                    args.output_dir, f'best_{args.best_metric}')
                checkpoint_file = os.path.join(
                    args.output_dir,
                    os.path.basename(os.readlink(checkpoint_file)))
            elif args.auto_resume:
                try:
                    utils.print_section_bar(
                        'Auto resume from the latest checkpoint')
                    if dist_utils.is_local_master():
                        logger.info(f'Checkpoint files {self.latest_cp_path}')
                    if os.path.exists(self.latest_cp_path):
                        checkpoint_file = self.latest_cp_path
                except Exception as e:
                    logger.info(f'[Error] {e}')
                    pass

        saved_state = None
        if checkpoint_file:
            assert os.path.exists(checkpoint_file), \
                f'model does not exist! {checkpoint_file}'
            if dist_utils.is_local_master():
                utils.print_section_bar('Restore from checkpoint')
                logger.info(f'Checkpoint files {checkpoint_file}')
            saved_state = checkpoint.load_states_from_checkpoint(checkpoint_file)
            options.set_encoder_params_from_state(
                saved_state.encoder_params, args)

        tokenizer = tfs.AutoTokenizer.from_pretrained(
            args.pretrained_model_cfg)
        tokenizer.add_special_tokens(
            {'additional_special_tokens': config.TOKENS})

        encoder = models.HFBertEncoder.init_encoder(args, len(tokenizer))
        reader = models.Reader(args, encoder)

        if args.inference_only:
            optimizer = None
        else:
            optimizer = model_utils.get_optimizer(
                reader,
                learning_rate=args.learning_rate,
                adam_eps=args.adam_eps,
                weight_decay=args.weight_decay)

        self.reader, self.optimizer = model_utils.setup_for_distributed_mode(
            reader,
            optimizer,
            args.device,
            args.n_gpu,
            args.local_rank,
            args.fp16,
            args.fp16_opt_level)

        self.start_epoch = 0
        self.start_offset = 0
        self.global_step = 0
        self.args = args
        
        if args.train_file is not None:
            self.topk_cp_info = self._load_topk_checkpoints_info()

        if saved_state:
            self._load_saved_state(saved_state)

        self.saved_state = saved_state
        self.tokenizer = tokenizer

    @property
    def latest_softlink(self):
        return os.path.join(self.args.output_dir, 'latest')

    @property
    def latest_cp_path(self):
        try:
            return os.path.join(
                self.args.output_dir,
                os.readlink(self.latest_softlink))
        except:
            pass
        return ''

    def get_train_dataloader(self, train_dataset, shuffle=True, offset=0):
        if torch.distributed.is_initialized():
            train_sampler = sampler.DistributedSampler(
                train_dataset,
                num_replicas=self.args.distributed_world_size,
                rank=self.args.local_rank,
                shuffle=shuffle)
            train_sampler.set_offset(offset)
        else:
            assert self.args.local_rank == -1
            train_sampler = torch.utils.data.RandomSampler(train_dataset)

        train_data_collator = data_collator.DataCollator(
            self.args.data_name,
            self.tokenizer,
            self.args.max_seq_len,
            self.args.max_num_answers,
            self.args.passages_per_question,
            self.args.special_attention,
            self.args.passage_attend_history,
            is_train=True,
            shuffle=True)

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=train_data_collator,
            drop_last=False)

        return dataloader

    def get_eval_data_loader(self, eval_dataset):
        if torch.distributed.is_initialized():
            eval_sampler = sampler.SequentialDistributedSampler(
                eval_dataset,
                num_replicas=self.args.distributed_world_size,
                rank=self.args.local_rank)
        else:
            assert self.args.local_rank == -1
            eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)

        eval_data_collator = data_collator.DataCollator(
            self.args.data_name,
            self.tokenizer,
            self.args.max_seq_len,
            self.args.max_num_answers,
            self.args.passages_per_question_predict,
            self.args.special_attention,
            self.args.passage_attend_history,
            is_train=False,
            shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.dev_batch_size,
            pin_memory=True,
            sampler=eval_sampler,
            num_workers=0,
            collate_fn=eval_data_collator,
            drop_last=False)

        return dataloader

    def run_train(self):
        args = self.args

        train_dataset = reader_dataset.ReaderDataset(args.train_file)
        train_dataloader = self.get_train_dataloader(
            train_dataset,
            shuffle=True,
            offset=self.start_offset)

        updates_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps)
        total_updates = updates_per_epoch * args.num_train_epochs

        dataloader_steps = self.start_offset // (
            args.distributed_world_size * args.batch_size)
        updated_steps = (dataloader_steps // 
                         args.gradient_accumulation_steps) + (
                             self.start_epoch * updates_per_epoch)
        remaining_updates = total_updates - updated_steps

        # global_step is added per dataloader step.
        calc_global_step = (self.start_epoch * len(train_dataloader) + 
                            dataloader_steps)

        assert self.global_step == calc_global_step, \
            (f'global step = {self.global_step}, '
             f'calc global step = {calc_global_step}')

        self.scheduler = model_utils.get_schedule_linear(
            self.optimizer,
            warmup_steps=args.warmup_steps,
            training_steps=total_updates,
            last_epoch=self.global_step-1)

        if self.saved_state:
            if self.saved_state.scheduler_dict:
                if dist_utils.is_local_master():
                    logger.info(f'Loading scheduler state ...')
                self.scheduler.load_state_dict(self.saved_state.scheduler_dict)

        utils.print_section_bar('Training')
        if dist_utils.is_local_master():
            logger.info(f'Total updates = {total_updates}')
            logger.info(
                f'Updates per epoch (/gradient accumulation) = '
                f'{updates_per_epoch}')
            logger.info(
                f'Steps per epoch (dataloader) = {len(train_dataloader)}')
            logger.info(
                f'Gradient accumulation steps = '
                f'{args.gradient_accumulation_steps}')
            logger.info(
                f'Start offset of the epoch {self.start_epoch} (dataset) = '
                f'step {self.start_offset}')
            logger.info(
                f'Updated step of the epoch {self.start_epoch} (dataloader) = '
                f'step {updated_steps}')
            logger.info(
                f'Total remaining updates = {remaining_updates}')

        # Starts training here.
        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            utils.print_section_bar(f'Epoch {epoch}')

            if isinstance(train_dataloader.sampler, sampler.DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            self._train_epoch(epoch, train_dataloader)

            if isinstance(train_dataloader.sampler, sampler.DistributedSampler):
                train_dataloader.sampler.set_offset(0)

        utils.print_section_bar('Training finished.')
        if dist_utils.is_local_master():
            best_em = -self.topk_cp_info['EM'][0][0]
            best_f1 = -self.topk_cp_info['F1'][0][0]
            best_em_path = self.topk_cp_info['EM'][0][1]
            best_f1_path = self.topk_cp_info['F1'][0][1]
            logger.info(f'Best EM {best_em * 100:.2f} path = {best_em_path}')
            logger.info(f'Best F1 {best_f1 * 100:.2f} path = {best_f1_path}')

        return

    def validate_and_save(self, epoch, offset):
        curr_em, curr_f1 = self.validate()

        args = self.args

        if dist_utils.is_local_master():
            cp_path = self._save_checkpoint(epoch, offset, dry_run=True)

            # Uses min heap, so add a negative
            if curr_em > -self.topk_cp_info['EM'][0][0]:
                logger.info(f'New best EM {curr_em*100:.2f} on dev')
                self._save_checkpoint(epoch, offset)
                utils.softlink(
                    cp_path,
                    os.path.join(args.output_dir, 'best_em'))

            if curr_f1 > -self.topk_cp_info['F1'][0][0]:
                logger.info(f'New best F1 {curr_f1*100:.2f} on dev')
                self._save_checkpoint(epoch, offset)
                utils.softlink(
                    cp_path,
                    os.path.join(args.output_dir, 'best_f1'))

            heapq.heappush(self.topk_cp_info['EM'], (-curr_em, cp_path))
            heapq.heappush(self.topk_cp_info['F1'], (-curr_f1, cp_path))
            tmp = []
            for _ in range(min(args.topk_em, len(self.topk_cp_info['EM']))):
                heapq.heappush(tmp, heapq.heappop(self.topk_cp_info['EM']))
            self.topk_cp_info['EM'] = tmp
            tmp = []
            for _ in range(min(args.topk_f1, len(self.topk_cp_info['F1']))):
                heapq.heappush(tmp, heapq.heappop(self.topk_cp_info['F1']))
            self.topk_cp_info['F1']= tmp

            best_em = -self.topk_cp_info['EM'][0][0]
            best_f1 = -self.topk_cp_info['F1'][0][0]
            best_em_path = self.topk_cp_info['EM'][0][1]
            best_f1_path = self.topk_cp_info['F1'][0][1]
            logger.info(f'Curr EM {curr_em * 100:.2f}')
            logger.info(f'Curr F1 {curr_f1 * 100:.2f}')
            logger.info(f'Best EM {best_em * 100:.2f} path = {best_em_path}')
            logger.info(f'Best F1 {best_f1 * 100:.2f} path = {best_f1_path}')

            self._save_topk_checkpoints_info()

            all_saved_cps = checkpoint.get_saved_checkpoints(
                args, args.checkpoint_filename_prefix)
            keep_cps = (set(c[1] for c in self.topk_cp_info['EM'])
                        | set(c[1] for c in self.topk_cp_info['F1']))
            keep_cps = set(
                [os.path.join(args.output_dir, cp) for cp in keep_cps])
            if self.latest_cp_path:
                keep_cps.update([self.latest_cp_path])
            for cp in all_saved_cps:
                if cp not in keep_cps: 
                    os.remove(cp)

    def validate(self):
        if dist_utils.is_local_master():
            logger.info('Validation ...')

        args = self.args
        topk_passages = self.topk_passages
        eval_dataset = reader_dataset.ReaderDataset(args.dev_file)
        eval_dataloader = self.get_eval_data_loader(eval_dataset)

        all_results = []
        validate_batch_times = []
        for step, batch in enumerate(eval_dataloader):
            self.reader.eval()
            step += 1

            if step % 100 == 0 and dist_utils.is_local_master():
                logger.info(
                    f'Eval step {step} / {len(eval_dataloader)}; '
                    f'eval time per batch = {np.mean(validate_batch_times):.2f}')

            batch = model_utils.move_to_device(batch, args.device)

            start_time = time.time()
            if args.local_rank != -1:
                # Uses DDP.
                with self.reader.no_sync(), torch.no_grad():
                    logits, _ = self.reader(
                        batch, self.global_step, end_task_only=True)
            else:
                # Uses a single GPU.
                with torch.no_grad():
                    logits, _ = self.reader(
                        batch, self.global_step, end_task_only=True)
            end_time = time.time()
            validate_batch_times.append(end_time - start_time)

            batch_predictions = self._get_best_prediction(
                logits['start'],
                logits['end'],
                logits['relevance'],
                batch['samples'])

            all_results.extend(batch_predictions)

            # Deletes output of the current iteration to save memory.
            del logits

        all_passage_f1s = []
        all_passage_at_k = []
        all_passage_em_at_k = []
        for pred in all_results:
            # we only have a single answer
            gold_answer = pred.gold_answers[0]

            passage_at_k = utils.convert_to_at_k(pred.passage_answers)
            passage_em_at_k = []
            for topk_passage_idx in range(topk_passages):

                spans = pred.passage_spans[topk_passage_idx]

                ems = []
                for s_i, s in enumerate(spans):
                    if s is not None:
                        em = eval.compute_exact(gold_answer, s.prediction_text)
                        em = {True: 1, False: 0}[em]
                        ems.append(em)
                    else:
                        # an empty span will be counted as an empty string
                        ems.append(0)

                if topk_passage_idx == 0:
                    f1 = 0.0
                    if spans and spans[0] is not None:
                        f1 = eval.compute_f1(
                            gold_answer, spans[0].prediction_text)
                    all_passage_f1s.append(f1)

                em_at_k = utils.convert_to_at_k(ems)
                passage_em_at_k.append(em_at_k)

            all_passage_at_k.append(passage_at_k)
            all_passage_em_at_k.append(passage_em_at_k)

        # Gathers results from other GPUs
        limit = len(eval_dataset)
        all_passage_at_k = torch.cat(
            dist_utils.all_gather(all_passage_at_k)).int().numpy()[:limit]
        all_passage_em_at_k = torch.cat(
            dist_utils.all_gather(all_passage_em_at_k)).int().numpy()[:limit]
        all_passage_f1s = torch.cat(
            dist_utils.all_gather(all_passage_f1s)).float().numpy()[:limit]

        avg_passage_at_k = np.ma.masked_where(
            all_passage_at_k == -1, all_passage_at_k).mean(axis=0).tolist()
        avg_passage_em_at_k = np.ma.masked_where(
            all_passage_em_at_k == -1, all_passage_em_at_k).mean(axis=0).tolist()
        avg_passage_f1 = float(np.mean(all_passage_f1s))

        if dist_utils.is_local_master():
            passage_at_k_str = ''
            passage_at_k_dic = {}
            for k_i, acc in enumerate(avg_passage_at_k):
                n = f'Passage@{k_i+1}'
                if acc is None:
                    passage_at_k_str += f'{n} = 0.0; '
                    passage_at_k_dic[n] = 0.0
                else:
                    passage_at_k_str += f'{n} = {acc * 100:.2f}; '
                    passage_at_k_dic[n] = acc

            passage_em_at_k_str = ''
            passage_em_at_k_dic = {}
            for p, em_at_k in enumerate(avg_passage_em_at_k):
                n_p = f'Rank {p+1} passage'
                passage_em_at_k_str += f'{n_p}: '
                for k_i, acc in enumerate(em_at_k):
                    n_em = f'EM@{k_i+1}'
                    passage_em_at_k_str += f'{n_em} = {acc * 100:.2f}; '
                    passage_em_at_k_dic[f'{n_p} {n_em}'] = acc
                passage_em_at_k_str += '\n'

            logger.info(f'eval_top_docs = {args.eval_top_docs[0]}')
            logger.info(f'F1 = {avg_passage_f1*100:.2f}')
            logger.info(passage_at_k_str)
            logger.info(passage_em_at_k_str)

            # Gathers numerical data from other GPUs.
            # Strings should be obtained directly from eval_dataset.
            if args.prediction_results_file:
                self._save_predictions(
                    args.prediction_results_file, all_results)

        return avg_passage_em_at_k[0][0], avg_passage_f1


    def _train_epoch(self, epoch, train_dataloader):
        args = self.args
        epoch_loss = 0
        rolling_train_losses = collections.defaultdict(int)
        rolling_train_others = collections.defaultdict(int)

        step_offset = 0
        # For restoring from a checkpoint.
        if train_dataloader.sampler.current_offset != 0:
            step_offset += (train_dataloader.sampler.current_offset // 
                            (args.distributed_world_size
                             * args.batch_size))

        train_batch_times = []
        start_time = time.time()
        for step, batch in enumerate(train_dataloader, start=step_offset):
            self.reader.train()
            step += 1

            batch_start_time = time.time()
            if step % args.gradient_accumulation_steps != 0 \
                    and args.local_rank != -1:
                with self.reader.no_sync():
                    losses, others = self._training_step(batch)
            else:
                losses, others = self._training_step(batch)
            batch_end_time = time.time()
            train_batch_times.append(batch_end_time - batch_start_time)

            self.global_step += 1

            # Saves latest checkpoint every X minutes.
            if dist_utils.is_local_master():
                now_time = time.time()
                time_diff = now_time - start_time
                # Converts seconds to minutes.
                if time_diff // \
                        (60 * args.save_checkpoint_every_minutes) == 1:
                    logger.info(
                        f'Save checkpoint every '
                        f'{args.save_checkpoint_every_minutes} minutes.')
                    dataset_offset = (step
                                      * args.distributed_world_size
                                      * args.batch_size)
                    cp_path = self._save_checkpoint(epoch, dataset_offset)
                    if self.latest_cp_path:
                        os.remove(self.latest_cp_path)
                    utils.softlink(cp_path, self.latest_softlink)
                    start_time = now_time

            '''
                record loss
            '''
            epoch_loss += losses['total']
            for k, loss in losses.items():
                rolling_train_losses[k] += loss
            for k, other in others.items():
                # other could be -1 if adv_loss not applicable
                rolling_train_others[k] += max(other, 0)

            '''
                parameters update
            '''
            if (step - step_offset) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), args.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.reader.parameters(), args.max_grad_norm
                        )
    
                self.scheduler.step()
                self.optimizer.step()
                self.reader.zero_grad()

            if self.global_step % args.log_batch_step == 0:
                lr = self.optimizer.param_groups[0]['lr']
                if dist_utils.is_local_master():
                    avg_batch_time = np.mean(train_batch_times)
                    logger.info(
                        f'Epoch: {epoch}: '
                        f'Step: {step}/{len(train_dataloader)}; '
                        f'Global_step={self.global_step}; '
                        f'lr={lr:.3e}; '
                        f'train time per batch = {avg_batch_time:.2f}')

            if (step - step_offset) % args.train_rolling_loss_step == 0:

                log_str = (f'Avg. loss and other in the recent '
                           f'{args.train_rolling_loss_step} batches: \n')
                for k, loss in rolling_train_losses.items():
                    loss /= args.train_rolling_loss_step
                    loss = torch.cat(
                        dist_utils.all_gather([loss])).mean().numpy()
                    log_str += ('    -' + f'{k:>21} loss: {loss:.4f}\n')

                for k, other in rolling_train_others.items():
                    other /= args.train_rolling_loss_step
                    other = torch.cat(
                        dist_utils.all_gather([other])).mean().numpy()
                    log_str += ('    -' + f'{k:>20} other: {other:.8f}\n')

                if dist_utils.is_local_master():
                    logger.info(f'Train: global step = {self.global_step}; '
                                f'step = {step}')
                    logger.info(log_str)

                rolling_train_losses = collections.defaultdict(int)
                rolling_train_others = collections.defaultdict(int)

            if self.global_step % args.eval_step == 0:
                if dist_utils.is_local_master():
                    logger.info(
                        f'Validation: Epoch: {epoch} '
                        f'Step: {step}/{len(train_dataloader)}')
                dataset_offset = (step
                                  * args.distributed_world_size
                                  * args.batch_size)
                self.validate_and_save(epoch, dataset_offset)

        epoch_loss = epoch_loss / len(train_dataloader)

        if dist_utils.is_local_master():
            logger.info(f'Avg. total Loss of epoch {epoch} ={epoch_loss:.3f}')

    def _save_topk_checkpoints_info(self):
        dic = {}
        dic['EM'] = [(-c[0], c[1])for c in self.topk_cp_info['EM']]
        dic['F1'] = [(-c[0], c[1])for c in self.topk_cp_info['F1']]
        path = os.path.join(
            self.args.output_dir, self.topk_cp_info_filename)
        with open(path, 'w') as f:
            json.dump(dic, f, indent=4)

    def _load_topk_checkpoints_info(self): 
        path = os.path.join(
            self.args.output_dir, self.topk_cp_info_filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                dic = json.load(f)
            dic['EM'] = [(-c[0], c[1])for c in dic['EM']]
            dic['F1'] = [(-c[0], c[1])for c in dic['F1']]
        else:
            dic = {}
            dic['EM'] = [(0, '')]
            dic['F1'] = [(0, '')]
        return dic

    def _save_checkpoint(self, epoch, offset, dry_run=False):
        cp_path = os.path.join(
            self.args.output_dir,
            '.'.join(
                [
                    self.args.checkpoint_filename_prefix,
                    str(epoch),
                    str(offset),
                    str(self.global_step),
                ]
            )
        )

        if dry_run:
            return os.path.basename(cp_path)
        # file already saved!
        if os.path.exists(cp_path):
            return os.path.basename(cp_path)

        logger.info(f'Saved checkpoint to {cp_path}')

        model_to_save = model_utils.get_model_obj(self.reader)
        meta_params = options.get_encoder_params_state(self.args)
        state = checkpoint.CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            amp.state_dict(),
            offset,
            epoch,
            self.global_step,
            meta_params,
        )

        torch.save(state._asdict(), cp_path)
        return os.path.basename(cp_path)

    def _load_saved_state(self, saved_state: checkpoint.CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        global_step = saved_state.global_step
        if offset == 0:  # epoch has been completed
            epoch += 1

        if dist_utils.is_local_master():
            logger.info(
                f'Loading checkpoint @'
                f'epoch = {epoch}, '
                f'offset = {offset}, '
                f'global_step = {global_step}, '
            )
        self.start_epoch = epoch
        self.start_offset = offset
        self.global_step = global_step

        model_to_load = model_utils.get_model_obj(self.reader)
        if saved_state.model_dict:
            if dist_utils.is_local_master():
                logger.info('Loading model weights from saved state ...')
            if self.args.train_file is None:
                model_to_load.load_state_dict(
                    saved_state.model_dict, strict=False)
            else:
                model_to_load.load_state_dict(saved_state.model_dict)


        if self.args.train_file is not None:
            if saved_state.optimizer_dict:
                if dist_utils.is_local_master():
                    logger.info('Loading saved optimizer state ...')
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

        if self.args.auto_resume:
            self.optimizer.state = {}
            amp.load_state_dict(saved_state.amp_dict)

    def _get_best_prediction(
        self,
        start_logits,
        end_logits,
        relevance_logits,
        samples_batch: List[data_class.ReaderSample]
    ) -> List[ReaderQuestionPredictions]:

        args = self.args
        topk_spans = self.topk_spans
        topk_passages = self.topk_passages
        passage_thresholds = self.args.eval_top_docs

        max_answer_length = args.max_answer_length
        questions_num, passages_per_question = relevance_logits.size()

        _, idxs = torch.sort(relevance_logits, dim=1, descending=True)

        batch_results = []
        max_num_passages = passage_thresholds[0]
        for q in range(questions_num):
            sample = samples_batch[q]

            non_empty_passages_num = len(sample.passages)

            passage_spans = []
            passage_answers = []
            for p in range(passages_per_question):

                # Needs topk passage but some passages will be passed because of
                # empty passages.
                if len(passage_spans) == topk_passages:
                    break

                passage_idx = idxs[q, p].item()

                if not (passage_idx < max_num_passages):
                    continue
                
                # Empty passage is selected, so skip.
                if passage_idx >= non_empty_passages_num:
                    continue
                
                reader_passage = sample.passages[passage_idx]
                sequence_ids = reader_passage.sequence_ids
                sequence_len = sequence_ids.size(0)
                reader_passage.has_answer
                # Assumes question & title information is at the beginning of the sequence

                p_start_logits = start_logits[q, passage_idx].tolist()
                p_end_logits = end_logits[q, passage_idx].tolist()
                best_spans = du.get_best_spans(
                    p_start_logits,
                    p_end_logits,
                    max_answer_length,
                    passage_idx,
                    reader_passage.span_texts,
                    reader_passage.span_types,
                    reader_passage.mask_cls.tolist(),
                    relevance_logits[q, passage_idx].item(),
                    top_spans=10)
                
                best_spans = best_spans[:topk_spans]
                best_spans += [None] * (topk_spans - len(best_spans))
                passage_spans.append(best_spans)
                assert len(passage_spans[-1]) == topk_spans

                passage_answers.append(
                    {True: 1, False: 0}[reader_passage.has_answer])

            # No passage
            # -1 as padding
            passage_answers += [-1] * (topk_passages - len(passage_answers))
            passage_spans += [[None]*topk_spans] * (topk_passages - len(passage_spans))
            assert len(passage_answers) == topk_passages

            batch_results.append(
                ReaderQuestionPredictions(
                    sample.id, sample.answers, passage_spans, passage_answers))
        return batch_results

    def _training_step(self, batch) -> torch.Tensor:
        args = self.args
        batch = model_utils.move_to_device(batch, args.device)
        logits, others = self.reader(batch, self.global_step)

        losses = models.compute_loss(args, logits, batch, others)

        losses = {k: loss.mean() for k, loss in losses.items()}

        if args.fp16:
            with amp.scale_loss(losses['total'], self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses['total'].backward()

        return {k: v.item() for k, v in losses.items()}, others

    def _get_preprocessed_filepaths(self, data_files: List, is_train: bool):
        serialized_files = [fn for fn in data_files if fn.endswith('.pkl')]
        if serialized_files:
            return serialized_files

        assert len(data_files) == 1, \
            'Only 1 source file pre-processing is supported.'

        # Data may have been serialized and cached before,
        # Tries to find ones from same dir.
        def _find_cached_files(path: str):
            dir_path, base_name = os.path.split(path)
            base_name = base_name.replace('.json', '')
            out_file_prefix = os.path.join(dir_path, base_name)
            out_file_pattern = out_file_prefix + '*.pkl'
            return glob.glob(out_file_pattern), out_file_prefix

        serialized_files, _ = _find_cached_files(data_files[0])

        assert serialized_files, 'run preprocessing code before training'

        if serialized_files:
            logger.info('Found preprocessed files. %s', serialized_files)
            return serialized_files

    def _save_predictions(
        self,
        out_file: str,
        prediction_results: List[ReaderQuestionPredictions]):

        logger.info(f'Saving prediction results to  {out_file}')

        with open(out_file, 'w', encoding='utf-8') as f:
            save_results = []
            for r in prediction_results:

                result = {'question': r.id, 'gold_answers': r.gold_answers}
                passage_preds = []
                for p_topk, (spans, p_ans) in enumerate(
                    zip(r.passage_spans, r.passage_answers)):
                    span_preds = []
                    for span_topk, span in enumerate(spans):
                        span_preds.append(
                            {
                                'span_topk': span_topk,
                                'prediction': {
                                    'text': (span.prediction_text 
                                             if span is not None else None),
                                    'score': (span.span_score
                                              if span is not None else None),
                                    'relevance_score': (
                                        span.relevance_score
                                        if span is not None else None),
                                }
                            })
                    passage_pred = {
                        'passage_topk': p_topk,
                        'passage_answer': p_ans,
                        'passage_idx': (spans[0].passage_index
                                        if spans[0] is not None else None),
                        'passage': (spans[0].passage_text
                                    if spans[0] is not None else None),
                        'spans': span_preds,
                    }
                    passage_preds.append(passage_pred)
                result['predictions'] = passage_preds 
                save_results.append(result)
            json.dump(save_results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()

    options.add_encoder_params(parser)
    options.add_f_div_regularization_params(parser)
    options.add_cuda_params(parser)
    options.add_training_params(parser)
    options.add_data_params(parser)
    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    if args.passage_attend_history:
        assert args.special_attention, \
            'passage_attend_history is a kind of special attention.'

    assert os.path.exists(args.pretrained_model_cfg), \
        (f'{args.pretrained_model_cfg} doesn\'t exist. '
         f'Please manually download the HuggingFace model.')
    options.setup_args_gpu(args)
    # Makes sure random seed is fixed.
    # set_seed must be called after setup_args_gpu.
    options.set_seed(args)

    if dist_utils.is_local_master():
        utils.print_args(args)

    trainer = ReaderTrainer(args)

    if args.train_file is not None:
        trainer.run_train()
    elif args.dev_file is not None:
        logger.info('No train files are specified. Run validation.')
        trainer.validate()
    else:
        logger.warning(
            'Neither train_file or (checkpoint_file & dev_file) parameters '
            'are specified. Nothing to do.')


if __name__ == '__main__':
    main()
