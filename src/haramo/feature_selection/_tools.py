from boruta import BorutaPy
import pandas as pd


class BorutaPyWrapper(BorutaPy):

    def fit(self, X, y):
        self._is_dataframe = isinstance(X, pd.DataFrame)
        return super().fit(X, y)

    def transform(self, X, weak=False):
        return self._transform(X, weak)

    def fit_transform(self, X, y, weak=False):
        self.fit(X, y)
        return self._transform(X, weak)

    def _transform(self, X, weak=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError("You need to call the fit(X, y) method first.")

        if weak:
            indices = self.support_ + self.support_weak_
        else:
            indices = self.support_

        if self._is_dataframe:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]

        return X
