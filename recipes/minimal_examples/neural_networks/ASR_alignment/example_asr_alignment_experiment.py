#!/usr/bin/python
import os
import speechbrain as sb
from speechbrain.alignment.aligner import ViterbiAligner
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.utils.train_logger import summarize_average
from speechbrain.utils.train_logger import summarize_error_rate

experiment_dir = os.path.dirname(os.path.realpath(__file__))
params_file = os.path.join(experiment_dir, "params.yaml")
data_folder = "../../../../samples/audio_samples/nn_training_samples"
data_folder = os.path.realpath(os.path.join(experiment_dir, data_folder))
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, {"data_folder": data_folder})

viterbi_aligner = ViterbiAligner()


class AlignBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)
        x = params.model(feats, init_params=init_params)
        x = params.lin(x, init_params)
        outputs = params.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        ids, phns, phn_lens = targets

        prev_alignments = viterbi_aligner.get_prev_alignments(
            ids, predictions, lens, phns, phn_lens
        )

        loss = params.compute_cost(predictions, prev_alignments)

        stats = {}

        if stage != "train":
            alignments, viterbi_scores = viterbi_aligner(
                predictions, lens, phns, phn_lens
            )

            viterbi_aligner.store_alignments(ids, alignments)

            phns = undo_padding(phns, phn_lens)
            stats["PER"] = wer_details_for_batch(ids, phns, alignments)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        print("Epoch %d complete" % epoch)
        print("Train loss: %.2f" % summarize_average(train_stats["loss"]))
        print("Valid loss: %.2f" % summarize_average(valid_stats["loss"]))
        print("Valid PER: %.2f" % summarize_error_rate(valid_stats["PER"]))

        print("Recalculating and recording alignments...")
        self.evaluate(train_set)


train_set = params.train_loader()
first_x, first_y = next(iter(train_set))
align_brain = AlignBrain(
    modules=[params.model, params.lin],
    optimizer=params.optimizer,
    first_inputs=[first_x],
)
align_brain.fit(range(params.N_epochs), train_set, params.valid_loader())
test_stats = align_brain.evaluate(params.test_loader())
print("Test PER: %.2f" % summarize_error_rate(test_stats["PER"]))


# Integration test: check that the model overfits the training data
def test_error():
    assert align_brain.avg_train_loss < 2.0
