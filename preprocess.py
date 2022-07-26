from sklearn.preprocessing import OneHotEncoder, Normalizer, LabelBinarizer
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd

class prep():
    def __init__(self):
        '''Returns label binarizer, mean imputer, normalizer, and one hot encoder'''
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.norm = Normalizer()
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.lbin = LabelBinarizer()

    def prep_ohe(self, df, categorical_cols, train=False):
        if train:
            cat_feats = self.enc.fit_transform(df[categorical_cols])
        else:
            cat_feats = self.enc.transform(df[categorical_cols])

        feat_names = list(self.enc.get_feature_names(categorical_cols))
        cat_feats = pd.DataFrame(cat_feats, columns=feat_names)
        return cat_feats

    def prep_impute_normalize(self, df, numerical_cols, train=False):
        if train:
            imputed = self.imp.fit_transform(df[numerical_cols])
            num_feats = self.norm.fit_transform(imputed)
        else:
            imputed = self.imp.transform(df[numerical_cols])
            num_feats = self.norm.transform(imputed)

        num_feats = pd.DataFrame(num_feats, columns=numerical_cols)
        return num_feats

    def prep_binary(self, df, binary_cols, train=False):
        if train:
            bin_feats = self.lbin.fit_transform(df[binary_cols])
        else:
            bin_feats = self.lbin.transform(df[binary_cols])

        return pd.DataFrame(bin_feats, columns=binary_cols)

    def prep_combine_feats(self, dfs):
        return pd.concat(dfs, axis=1)

    def preprocess(self, df, numerical_cols, categorical_cols, binary_cols, target_col, train=False):
        cat_feats = self.prep_ohe(df, categorical_cols, train)
        print("Categorical Feats done")
        num_feats = self.prep_impute_normalize(df, numerical_cols, train)
        print("Numerical Feats done")
        bin_feats = self.prep_binary(df, binary_cols, train)
        print("Binary Feats done")

        prepared_data = self.prep_combine_feats([num_feats, cat_feats, bin_feats])

        prepared_data[target_col] = df[target_col]

        return prepared_data