import os
import pandas as pd
import tempeh.configurations as tc

from aif360.datasets import StandardDataset


default_mappings = {
    'label_maps': [{1.0: 'Passed', 0.0: 'Failed_or_not_attempted'}],
    'protected_attribute_maps': [{0.0: 'Male', 1.0: 'Female'},
                                 {0.0: 'White', 1.0: 'Black'}]
}

def default_preprocessing(df):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    return df[df.race.isin(['White','Black'])]


class LSACDataset(StandardDataset):
    """Law School GPA dataset.
    See https://github.com/microsoft/tempeh for details.
    """

    def __init__(self, label_name='pass_bar',
                 favorable_classes=[0],
                 protected_attribute_names=['sex', 'race'],
                 privileged_classes=[['Male'], ['White']],
                 instance_weights_name=None,
                 categorical_features=[],
                 na_values=[], custom_preprocessing=default_preprocessing,
                 metadata=default_mappings):
        
        dataset = tc.datasets["lawschool_gpa"]()
        X_train,X_test = dataset.get_X(format=pd.DataFrame)
        y_train, y_test = dataset.get_y(format=pd.Series)
        A_train, A_test = dataset.get_sensitive_features(name='race',
                                                         format=pd.Series)
        all_train = pd.concat([X_train, y_train, A_train], axis=1)
        all_test = pd.concat([X_test, y_test, A_test], axis=1)

        df = pd.concat([all_train, all_test], axis=0)

        super(LSACDataset, self).__init__(df=df,
            label_name=label_name,
            protected_attribute_names=protected_attribute_names,
            favorable_classes=favorable_classes,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)