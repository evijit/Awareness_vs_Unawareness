import os
from enum import Enum
from typing import Union

from aif360.datasets import StandardDataset
from hydra.utils import get_original_cwd
from LSACDataset import LSACDataset


def load_dataset(name):
    if name == "uci_adult":
        from aif360.datasets import AdultDataset
        return AdultDataset()
    if name == "compas":
        from aif360.datasets import CompasDataset
        return CompasDataset()
    if name == "law_school":
#         from aif360.datasets import LawSchoolGPADataset
        return LSACDataset()


def model_name2enum(name: str):
    return {
        "AdversarialDebiasing": DemographicAware.Adversarial_Debiasing,
        "VariationalFairAutoEncoder": DemographicAware.Variational_Fair_AutoEncoder,
        "MetaFairClassifier": DemographicAware.Meta_Fair_Classifier,
        "PrejudiceRemover": DemographicAware.Prejudice_Remover,
        "GridSearchReduction": DemographicAware.Grid_Search_Reduction,
        "AdversarialReweighting": DemographicFree.Adversarial_Reweighting,
        "GerryFairClassifier": DemographicFree.Gerry_Fair_Classifier,
        "RepeatedLossMinimization": DemographicFree.Repeated_Loss_Minimization,
    }[name]


class DemographicAware(Enum):
    Adversarial_Debiasing = "AdversarialDebiasing"
    Variational_Fair_AutoEncoder = "VariationalFairAutoEncoder"  # Not implemented
    Meta_Fair_Classifier = "MetaFairClassifier"
    Prejudice_Remover = "PrejudiceRemover"
    Grid_Search_Reduction = "GridSearchReduction"


class DemographicFree(Enum):
    Adversarial_Reweighting = "AdversarialReweighting"
    Gerry_Fair_Classifier = "GerryFairClassifier"
    Repeated_Loss_Minimization = "RepeatedLossMinimization"


FairAlgorithms = Union[DemographicAware, DemographicFree]


def get_adv_rew_dataset_input_base(dataset_name):
    if dataset_name == "uci_adult":
        from group_agnostic_fairness.data_utils.uci_adult_input import UCIAdultInput
        return UCIAdultInput(
            os.path.join(get_original_cwd(), 'data', 'raw', dataset_name),  # dataset_base_dir
            train_file=os.path.join(get_original_cwd(), 'data', 'raw', dataset_name, 'adult.data'),
            test_file=os.path.join(get_original_cwd(), 'data', 'raw', dataset_name, 'adult.test')
        )
    if dataset_name == "compas":
        from group_agnostic_fairness.data_utils.compas_input import CompasInput
        return CompasInput(
            os.path.join(get_original_cwd(), 'data', 'raw', dataset_name),  # dataset_base_dir
        )
    if dataset_name == 'law_school':
        from group_agnostic_fairness.data_utils.law_school_input import LawSchoolInput
        return LawSchoolInput(
            os.path.join(get_original_cwd(), 'data', 'raw', dataset_name)
        )
    raise Exception("Unknown dataset")


def create_model(model_name: str, dataset_name: str, dataset: StandardDataset, estimator=None, **kwargs):
    name: FairAlgorithms = model_name2enum(model_name)
    sess = kwargs.pop('sess')  # For TF based models, some don't use it
    if name == DemographicAware.Adversarial_Debiasing:
        from aif360.algorithms.inprocessing import AdversarialDebiasing
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

        protected_attribute = dataset.protected_attribute_names
        privileged_protected_attributes = dataset.privileged_protected_attributes
        unprivileged_protected_attributes = dataset.unprivileged_protected_attributes
        privileged_groups = tuple([{
            attr_name: privileged_value.item()
            for attr_name, privileged_value in zip(protected_attribute, privileged_protected_attributes)
        }])
        unprivileged_groups = tuple([{
            attr_name: unprivileged_value.item()
            for attr_name, unprivileged_value in zip(protected_attribute, unprivileged_protected_attributes)
        }])

        return AdversarialDebiasing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            scope_name='debiased_classifier',  # NOTE: Or use 'plain_classifier' for a non-debiased!
            debias=True,
            sess=sess,
            **kwargs
        )
    if name == DemographicAware.Variational_Fair_AutoEncoder:
        pass  # TODO: implement
    if name == DemographicAware.Meta_Fair_Classifier:
        from aif360.algorithms.inprocessing import MetaFairClassifier
        return MetaFairClassifier(
            **kwargs
        )
    if name == DemographicAware.Prejudice_Remover:
        from aif360.algorithms.inprocessing import PrejudiceRemover
        return PrejudiceRemover(
            **kwargs
        )
    if name == DemographicAware.Grid_Search_Reduction:
        from aif360.algorithms.inprocessing import GridSearchReduction
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression() if estimator is None else estimator

        constraints = kwargs.pop("constraint_class")
        return GridSearchReduction(
            estimator,
            constraints,
            **kwargs
        )
    if name == DemographicFree.Adversarial_Reweighting:
        from adversarial_reweighting_adapter import AdversarialReweightingAdapter
        from adversarial_reweighting_adapter import AdapterAdvRewInput

        feature_columns = dataset.feature_names
        label_column_name = dataset.label_names[0]
        model_dir = os.path.join(os.getcwd(), 'adversarial_reweighting_adapter')
        dataset_input = AdapterAdvRewInput(
            get_adv_rew_dataset_input_base(dataset_name),
        )

        return AdversarialReweightingAdapter(
            feature_columns,
            label_column_name,
            model_dir,
            dataset_input,
            **kwargs
        )
    if name == DemographicFree.Gerry_Fair_Classifier:
        from aif360.algorithms.inprocessing import GerryFairClassifier
        return GerryFairClassifier(
            **kwargs
        )
    if name == DemographicFree.Repeated_Loss_Minimization:
        pass  # TODO: implement
    raise ValueError("Unknown algorithm")