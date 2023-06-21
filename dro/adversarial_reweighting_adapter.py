import json
import os
import sys

import numpy as np
from aif360.algorithms import Transformer
import tensorflow as tf

from group_agnostic_fairness.AdvRewInputBase import AdvRewInputBase
from group_agnostic_fairness.fairness_metrics import RobustFairnessMetrics
from group_agnostic_fairness.main_trainer import get_estimator, write_to_output_file
from absl import flags

FLAGS = flags.FLAGS

IPS_WITH_LABEL_TARGET_COLUMN_NAME = "IPS_example_weights_with_label"
IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME = "IPS_example_weights_without_label"
SUBGROUP_TARGET_COLUMN_NAME = "subgroup"


class AdversarialReweightingAdapter(Transformer):
    """
    Adversarial Reweighting is an in-processing algorithm based on the 
    Fairness without Demographics through Adversarially Reweighted Learning paper by Lahoti et al.
    This class is an adapter for the AIF360 API to work with the code published with said paper at 
    https://github.com/google-research/google-research/tree/master/group_agnostic_fairness.
    """

    @staticmethod
    def __set_flags():
        # Flags for creating and running a model
        flags.DEFINE_string(
            "model_name",
            "adversarial_reweighting",
            "Name of the model to run"
        )
        flags.DEFINE_string(
            "base_dir",
            "/tmp",
            "Base directory for output."
        )
        flags.DEFINE_string(
            "model_dir",
            None,
            "Model directory for output."
        )
        flags.DEFINE_string(
            "output_file_name",
            "results.txt",
            "Output file where to write metrics to."
        )
        flags.DEFINE_string(
            "print_dir",
            None,
            "directory for tf.print output_stream."
        )

        # Flags for training and evaluation
        flags.DEFINE_integer(
            "total_train_steps",
            1280000,
            "Number of training steps."
        )
        flags.DEFINE_integer(
            "test_steps",
            1000,
            "Number of evaluation steps."
        )
        flags.DEFINE_integer(
            "min_eval_frequency",
            1000,
            "How often (steps) to run evaluation."
        )

        # Flags for loading dataset
        flags.DEFINE_string(
            "dataset_base_dir",
            "./group_agnostic_fairness/data/toy_data",
            "(string) path to dataset directory"
        )
        flags.DEFINE_string(
            "dataset",
            "uci_adult", "Name of the dataset to run"
        )
        flags.DEFINE_multi_string(
            "train_file",
            ["./group_agnostic_fairness/data/toy_data/train.csv"],
            "List of (string) path(s) to training file(s)."
        )
        flags.DEFINE_multi_string(
            "test_file",
            ["./group_agnostic_fairness/data/toy_data/test.csv"],
            "List of (string) path(s) to evaluation file(s)."
        )

        # # If the model has an adversary, the features for adversary are constructed
        # # in the corresponding custom estimator implementation by filtering feature_columns passed to the learner.
        flags.DEFINE_bool(
            "include_sensitive_columns",
            False,
            "Set the flag to include protected features in the feature_columns of the learner."
        )

        # Flags for setting common model parameters for all approaches
        flags.DEFINE_multi_integer(
            "primary_hidden_units",
            [64, 32],
            "Hidden layer sizes of main learner."
        )
        flags.DEFINE_integer(
            "embedding_dimension",
            32,
            "Embedding size; if 0, use one hot."
        )
        flags.DEFINE_integer(
            "batch_size",
            32,
            "Batch size."
        )
        flags.DEFINE_float(
            "primary_learning_rate",
            0.001,
            "learning rate for main learner."
        )
        flags.DEFINE_string(
            "optimizer",
            "Adagrad",
            "Name of the optimizer to use."
        )
        flags.DEFINE_string(
            "activation",
            "relu",
            "Name of the activation to use."
        )

        # # Flags for approaches that have an adversary
        # # Currently only for ''adversarial_reweighting'' Model.
        flags.DEFINE_multi_integer(
            "adversary_hidden_units",
            [32],
            "Hidden layer sizes of adversary."
        )
        flags.DEFINE_float(
            "adversary_learning_rate",
            0.001,
            "learning rate for adversary."
        )

        # # Flags for adversarial_reweighting model
        flags.DEFINE_string(
            "adversary_loss_type",
            "ce_loss",
            "Type of adversary loss function to be used. Takes values in [``ce_loss'',''hinge_loss'']. ``ce loss`` stands for cross-entropy loss"
        )
        flags.DEFINE_bool(
            "upweight_positive_instance_only",
            False,
            "Set the flag to weight only positive examples if in adversarial loss. Only used when adversary_loss_type parameter of adversarial_reweighting model is set to hinge_loss"
        )
        flags.DEFINE_bool(
            "adversary_include_label",
            True,
            "Set the flag to add label as a feature to adversary in the model."
        )
        flags.DEFINE_integer(
            "pretrain_steps",
            250,
            "Number of steps to train primary before alternating with adversary."
        )

        # # Flags for inverse_propensity_weighting Model
        flags.DEFINE_string(
            "reweighting_type",
            "IPS_without_label",
            "Type of reweighting to be performed. Takes values in [''IPS_with_label'', ''IPS_without_label'']"
        )

        TENSORFLOW_BOARD_BINARY = "/Users/pablokvitca/opt/miniconda3/envs/fairml-demo-free-comp/bin/tensorboard"
        tf.logging.set_verbosity(tf.logging.INFO)

    def __init__(
            self,
            feature_columns,
            label_column_name,
            model_dir,
            dataset_input: AdvRewInputBase,
            print_dir=None
    ):
        """
        Initializes the Adapter for the Adversarial Reweighting estimator
        """
        FLAGS(sys.argv)
        # AdversarialReweightingAdapter.__set_flags()
        self.dataset_input = dataset_input
        self.model_dir = model_dir

        _feature_columns, _, self.protected_groups, _label_column_name = (
            dataset_input.get_feature_columns(
                embedding_dimension=FLAGS.embedding_dimension,
                include_sensitive_columns=FLAGS.include_sensitive_columns
            )
        )
        # assert set(feature_columns) == set(_feature_columns), \
        #     f"Given feature columns {feature_columns} and given dataset input feature columns {_feature_columns} don't match"
        self.feature_columns = _feature_columns
        # assert label_column_name == _label_column_name, \
        #     f"Given label column name {label_column_name} and given dataset input label column name {_label_column_name} don't match"
        self.label_column_name = _label_column_name

        # Constructs a int list enumerating the number of subgroups in the dataset.
        # # For example, if the dataset has two (binary) protected_groups. The dataset has 2^2 = 4 subgroups, which we enumerate as [0, 1, 2, 3].
        # # If the  dataset has two protected features ["race","sex"] that are cast as binary features race=["White"(0), "Black"(1)], and sex=["Male"(0), "Female"(1)].
        # # We call their catesian product ["White Male" (00), "White Female" (01), "Black Male"(10), "Black Female"(11)] as subgroups  which are enumerated as [0, 1, 2, 3].
        self.subgroups = np.arange(
            len(self.protected_groups) *
            2)  # Assumes each protected_group has two possible values.

        # Adds additional fairness metrics
        fairness_metrics = RobustFairnessMetrics(
            label_column_name=label_column_name,
            protected_groups=self.protected_groups,
            subgroups=self.subgroups,
            print_dir=print_dir
        )
        eval_metrics_fn = fairness_metrics.create_fairness_metrics_fn()

        self.estimator = get_estimator(
            self.model_dir,
            "adversarial_reweighting",
            self.feature_columns,
            self.label_column_name
        )
        self.estimator = tf.estimator.add_metrics(self.estimator, eval_metrics_fn)

        self.train_steps = int(FLAGS.total_train_steps / FLAGS.batch_size)

    def _make_input_fn(self, dataset, batch_size=32):
        _input = AdapterAdvRewInput(self.dataset_input)
        return _input.get_input_fn(dataset, batch_size=batch_size)

    def transform(self, dataset):
        return dataset

    def fit(self, dataset, batch_size=32):
        train_spec = tf.estimator.TrainSpec(
            input_fn=self._make_input_fn(dataset, batch_size=batch_size),
            max_steps=self.train_steps
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self._make_input_fn(dataset, batch_size=batch_size),
            steps=FLAGS.test_steps
        )  # NOTE: eval will be also train data here!
        tf.estimator.train_and_evaluate(
            self.estimator,
            train_spec,
            eval_spec
        )

    def predict(self, dataset):
        eval_results = self.estimator.evaluate(
            input_fn=self._make_input_fn(dataset),
            steps=FLAGS.test_steps
        )
        eval_results_path = os.path.join(self.model_dir, FLAGS.output_file_name)
        write_to_output_file(eval_results, eval_results_path)
        pass  # TODO: return preds


class AdapterAdvRewInput(AdvRewInputBase):
    """
    AdvRewInput implementation to work with AIF360 adapter.
    """

    def __init__(self, base_input: AdvRewInputBase):
        """
        TODO: docs
        """
        self.base_input = base_input

    def get_input_fn(self, dataset, batch_size=128):
        """
        TODO: docs
        """

        def _input_fn():
            """Input_fn for the dataset."""

            # Extracts basic features and targets from filename_queue
            features, targets = self.extract_features_and_targets_from_dataset(dataset, batch_size)

            # Adds subgroup information to targets. Used to plot metrics.
            targets = self._add_subgroups_to_targets(features, targets)

            # Adds ips_example_weights to targets
            targets = self._add_ips_example_weights_to_targets(targets)

            # Unused in robust_learning models. Adding it for min-diff approaches.
            # Adding instance weight to features.
            features[self.weight_column_name] = tf.ones_like(
                targets[self.target_column_name], dtype=tf.float64)

            return features, targets

        return _input_fn

    def extract_features_and_targets_from_dataset(self, dataset, batch_size, favorable_label_value=1):
        """
        TOOD: docs
        """
        features = {
            col_name: dataset.features[:, dataset.feature_names.index(col_name)]
            for col_name in dataset.feature_names
        }
        features[self.base_input.target_column_name] = dataset.labels
        # features = self._binarize_protected_features(features)
        try:
            features = tf.train.batch(features, batch_size)
        except ValueError as err:
            print(err)

        targets = {
            self.base_input.target_column_name: tf.reshape(
                tf.cast(
                    tf.equal(
                        features.pop(self.base_input.target_column_name),
                        favorable_label_value
                    ),
                    tf.float64
                ),
                [-1, 1]
            )
        }
        targets[dataset.label_names[0]] = targets[self.base_input.target_column_name]
        return features, targets

    def extract_features_and_targets(self, filename_queue, batch_size):
        raise NotImplementedError()

    def _binarize_protected_features(self, features):
        """Processes protected features and binarize them."""
        for sensitive_column_name, sensitive_column_value in zip(
                self.base_input.sensitive_column_names, self.base_input.sensitive_column_values):
            features[sensitive_column_name] = tf.cast(
                tf.equal(
                    features.pop(sensitive_column_name), sensitive_column_value),
                tf.float64)
        return features

    def _add_subgroups_to_targets(self, features, targets):
        """Adds subgroup information to targets dictionary."""
        for sensitive_column_name in self.base_input.sensitive_column_names:
            targets[sensitive_column_name] = tf.reshape(
                tf.identity(features[sensitive_column_name]), [-1, 1])
        return targets

    def _add_ips_example_weights_to_targets(self, targets):
        """Add ips_example_weights to targets. Used in ips baseline model."""

        # Add subgroup information to targets
        target_subgroups = (targets[self.base_input.target_column_name],
                            targets[self.base_input.sensitive_column_names[1]],
                            targets[self.base_input.sensitive_column_names[0]])
        targets[SUBGROUP_TARGET_COLUMN_NAME] = tf.map_fn(
            lambda x: (2 * x[1]) + (1 * x[2]), target_subgroups, dtype=tf.float64)

        # Load precomputed IPS weights into a HashTable.
        ips_with_label_table = AdvRewInputBase._load_json_dict_into_hashtable(
            self.base_input.ips_with_label_file)
        ips_without_label_table = AdvRewInputBase._load_json_dict_into_hashtable(
            self.base_input.ips_without_label_file)

        # Adding IPS example weights to targets
        targets[IPS_WITH_LABEL_TARGET_COLUMN_NAME] = tf.map_fn(
            lambda x: ips_with_label_table.lookup(
                tf.cast((4 * x[0]) + (2 * x[1]) + (1 * x[2]), dtype=tf.int64)),
            target_subgroups,
            dtype=tf.float64)
        targets[IPS_WITHOUT_LABEL_TARGET_COLUMN_NAME] = tf.map_fn(
            lambda x: ips_without_label_table.lookup(
                tf.cast((2 * x[1]) + (1 * x[2]), dtype=tf.int64)),
            target_subgroups,
            dtype=tf.float64)

        return targets

    def get_feature_columns(self, embedding_dimension=0, include_sensitive_columns=True):
        """
        TODO: docs
        """
        with tf.gfile.Open(self.base_input.mean_std_file, "r") as mean_std_file:
            mean_std_dict = json.load(mean_std_file)
        with tf.gfile.Open(self.base_input.vocabulary_file, "r") as vocabulary_file:
            vocab_dict = json.load(vocabulary_file)

        feature_columns = []
        for i in range(0, len(self.base_input.feature_names)):
            if (self.base_input.feature_names[i] in [
                self.base_input.weight_column_name, self.base_input.target_column_name
            ]):
                continue
            elif self.base_input.feature_names[i] in self.base_input.sensitive_column_names:
                if include_sensitive_columns:
                    feature_columns.append(
                        tf.feature_column.numeric_column(self.base_input.feature_names[i]))
                else:
                    continue
            elif self.base_input.RECORD_DEFAULTS[i][0] == "?":
                sparse_column = tf.feature_column.categorical_column_with_vocabulary_list(
                    self.base_input.feature_names[i], vocab_dict[self.base_input.feature_names[i]])
                if embedding_dimension > 0:
                    feature_columns.append(
                        tf.feature_column.embedding_column(sparse_column,
                                                           embedding_dimension))
                else:
                    feature_columns.append(
                        tf.feature_column.indicator_column(sparse_column))
            else:
                mean, std = mean_std_dict[self.base_input.feature_names[i]]
                feature_columns.append(
                    tf.feature_column.numeric_column(
                        self.base_input.feature_names[i],
                        normalizer_fn=(lambda x, m=mean, s=std: (x - m) / s)))
        return feature_columns, \
               self.base_input.weight_column_name, \
               self.base_input.sensitive_column_names, \
               self.base_input.target_column_name

    def get_sensitive_column_names(self):
        return self.base_input.sensitive_column_names

    def set_sensitive_column_names(self, sensitive_column_names):
        self.base_input.sensitive_column_names = sensitive_column_names

    def del_sensitive_column_names(self):
        del self.base_input.sensitive_column_names

    sensitive_column_names = property(get_sensitive_column_names, set_sensitive_column_names, del_sensitive_column_names)

    def get_sensitive_column_values(self):
        return self.base_input.sensitive_column_values

    def set_sensitive_column_values(self, sensitive_column_values):
        self.base_input.sensitive_column_values = sensitive_column_values

    def del_sensitive_column_values(self):
        del self.base_input.sensitive_column_values

    sensitive_column_values = property(get_sensitive_column_values, set_sensitive_column_values, del_sensitive_column_values)

    def get_target_column_name(self):
        return self.base_input.target_column_name

    def set_target_column_name(self, target_column_name):
        self.base_input.target_column_name = target_column_name

    def del_target_column_name(self):
        del self.base_input.target_column_name

    target_column_name = property(get_target_column_name, set_target_column_name, del_target_column_name)

    def get_target_column_positive_value(self):
        return self.base_input.target_column_positive_value

    def set_target_column_positive_value(self, target_column_positive_value):
        self.base_input.target_column_positive_value = target_column_positive_value

    def del_target_column_positive_value(self):
        del self.base_input.target_column_positive_value

    target_column_positive_value = property(get_target_column_positive_value, set_target_column_positive_value, del_target_column_positive_value)

    def get_ips_with_label_file(self):
        return self.base_input.ips_with_label_file

    def set_ips_with_label_file(self, ips_with_label_file):
        self.base_input.ips_with_label_file = ips_with_label_file

    def del_ips_with_label_file(self):
        del self.base_input.ips_with_label_file

    ips_with_label_file = property(get_ips_with_label_file, set_ips_with_label_file, del_ips_with_label_file)

    def get_ips_without_label_file(self):
        return self.base_input.ips_without_label_file

    def set_ips_without_label_file(self, ips_without_label_file):
        self.base_input.ips_without_label_file = ips_without_label_file

    def del_ips_without_label_file(self):
        del self.base_input.ips_without_label_file

    ips_without_label_file = property(get_ips_without_label_file, set_ips_without_label_file, del_ips_without_label_file)

    def get_mean_std_file(self):
        return self.base_input.mean_std_file

    def set_mean_std_file(self, mean_std_file):
        self.base_input.mean_std_file = mean_std_file

    def del_mean_std_file(self):
        del self.base_input.mean_std_file

    mean_std_file = property(get_mean_std_file, set_mean_std_file, del_mean_std_file)

    def get_vocabulary_file(self):
        return self.base_input.vocabulary_file

    def set_vocabulary_file(self, vocabulary_file):
        self.base_input.vocabulary_file = vocabulary_file

    def del_vocabulary_file(self):
        del self.base_input.vocabulary_file

    vocabulary_file = property(get_vocabulary_file, set_vocabulary_file, del_vocabulary_file)

    def get_weight_column_name(self):
        return self.base_input.weight_column_name

    def set_weight_column_name(self, weight_column_name):
        self.base_input.weight_column_name = weight_column_name

    def del_weight_column_name(self):
        del self.base_input.weight_column_name

    weight_column_name = property(get_weight_column_name, set_weight_column_name, del_weight_column_name)