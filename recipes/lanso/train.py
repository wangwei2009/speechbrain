#!/usr/bin/python3
"""Recipe for training a classifier using the
mobvoihotwords Dataset.

To run this recipe, use the following command:
> python train.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/xvect.yaml (xvector system)

"""
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from torch.nn.utils import clip_grad_norm_
import numpy as np
from sklearn.metrics import confusion_matrix
from pcen.pcen import pcen
import pcen
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

global output_array
global label_array
output_array=np.array([])
label_array = np.array([])
class SpeakerBrain(sb.core.Brain):
    """Class for GSC training"
    """

    def __init__(  # noqa: C901
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        super(SpeakerBrain, self).__init__(  # noqa: C901
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )

        self.output_array=np.array([])
        self.label_array = np.array([])

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + command classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN and self.hparams.apply_data_augmentation:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # print("wavs.size():{}".format(wavs.size()))
        # print("lens.size():{}".format(lens.size()))
        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)

        feats = self.modules.mean_var_norm(feats, lens)
        # print("feats.size():{}".format(feats.size()))

        if stage == sb.Stage.TRAIN and self.hparams.apply_data_augmentation:
            if hasattr(self.hparams, "augment_spec"):
                feats = self.hparams.augment_spec(feats)

        # Embeddings + classifier
        outputs = self.modules.embedding_model(feats)
        if "classifier" in self.modules.keys():
            outputs = self.modules.classifier(outputs)
        # print("outputs.size():{}".format(outputs.size()))

        # Ecapa model uses softmax outside of its classifer
        if "softmax" in self.modules.keys():
            outputs = self.modules.softmax(outputs)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using command-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        command, _ = batch.command_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and self.hparams.apply_data_augmentation:
            command = torch.cat([command] * self.n_augment, dim=0)
        # print("command.size():{}".format(command.size()))

        if hasattr(self,"hparams.clip_grad"):
        # if True:
            norm = clip_grad_norm_(self.modules.embedding_model.parameters(), self.hparams.clip_grad)

        # compute the cost function
        loss = self.hparams.compute_cost(predictions, command, lens)
        # loss = sb.nnet.losses.nll_loss(predictions, command, lens)


        if stage != sb.Stage.TRAIN:
            output_label = torch.argmax(predictions[:, 0, :], dim=1).cpu().numpy()
            # label_list.append(command.cpu().numpy())
            self.label_array = np.concatenate((self.label_array, command.cpu().numpy()[:, 0]))
            self.output_array = np.concatenate((self.output_array, output_label))


        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, command, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

            self.output_array=np.array([])
            self.label_array = np.array([])

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

            if stage == sb.Stage.VALID:
                self.valid_stats = stage_stats
            if stage == sb.Stage.TEST:
                self.test_stats = stage_stats

            cm = confusion_matrix(self.label_array, self.output_array)
            print(cm)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

            self.valid_stats = {
                "loss": stage_stats['loss'],
                "ErrorRate": stage_stats["ErrorRate"],
            }
        if stage == sb.Stage.TEST:

            if self.hparams.use_tensorboard:
                test_stats = {
                    "loss": stage_stats['loss'],
                    "ErrorRate": stage_stats["ErrorRate"],
                }
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch}, self.train_stats, self.valid_stats, test_stats
                )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        test_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
        test_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not isinstance(train_set, DataLoader):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not isinstance(valid_set, DataLoader):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        if test_set is not None and not isinstance(test_set, DataLoader):
            test_set = self.make_dataloader(
                test_set,
                stage=sb.Stage.TEST,
                ckpt_prefix=None,
                **test_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:

            # Training stage
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        run_on_main(self._save_intra_epoch_ckpt)
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[sb.Stage.VALID, avg_valid_loss, epoch],
                    )

            # Test stage
            if test_set is not None:
                self.on_stage_start(sb.Stage.TEST, epoch)
                self.modules.eval()
                avg_test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        test_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                        avg_test_loss = self.update_average(
                            loss, avg_test_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[sb.Stage.TEST, avg_test_loss, epoch],
                    )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )                                                       # [channel, time]
        # sig = sig.transpose(0, 1).squeeze(1)                                            # channel-1
        sig = sig[0]                                            # channel-1
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("command")
    @sb.utils.data_pipeline.provides("command", "command_encoded")
    def label_pipeline(command):
        yield command
        command_encoded = label_encoder.encode_sequence_torch([command])
        yield command_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="command",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "command_encoded"]
    )

    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing GSC and annotation into csv files)
    from prepare_kws import prepare_kws

    # Data preparation
    run_on_main(
        prepare_kws,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        test_set=test_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = speaker_brain.evaluate(
        test_set=test_data,
        min_key="ErrorRate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
