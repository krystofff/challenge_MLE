import pandas as pd
from sklearn.linear_model import LogisticRegression

from typing import Any, Dict, Tuple, Union, List
from .constants import FEATURES_COLS, TARGET_COL


class DelayModel:

    
    def __init__(
        self,
        decision_threshold: float = 0.60,
        model_params: Dict[str, Any] | None = None,
    ):
        self._decision_threshold = float(decision_threshold)
        self._model: LogisticRegression | None = None
        self._model_params = model_params or {}

    @staticmethod
    def _compute_delay(df: pd.DataFrame) -> pd.Series:
        """delay = 1 if (Fecha-O - Fecha-I) > 15 minutes, else 0."""
        fi = pd.to_datetime(df["Fecha-I"])
        fo = pd.to_datetime(df["Fecha-O"])
        mins = (fo - fi).dt.total_seconds() / 60.0
        return (mins > 15).astype(int)
    
    def _one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df[["OPERA", "TIPOVUELO", "MES"]].copy()
        base["MES"] = base["MES"].astype(int)

        dummies = pd.get_dummies(
            base,
            columns=["OPERA", "TIPOVUELO", "MES"],
            prefix=["OPERA", "TIPOVUELO", "MES"],
            drop_first=True,
        )

        for col in FEATURES_COLS:
            if col not in dummies.columns:
                dummies[col] = 0

        return dummies[list(FEATURES_COLS)].astype(int)

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        df = data.copy()
        target = None

        if target_column is not None:
            if target_column not in df.columns:
                df[target_column] = self._compute_delay(df)
            target = df[[target_column]].copy()

        features = self._one_hot(df)
        return (features, target) if target_column is not None else features


    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        y = target[TARGET_COL[0]].astype(int).values
        n = len(y)
        if n == 0:
            raise ValueError("Empty training target.")
        
        n1 = int((y == 1).sum())
        n0 = n - n1
        
        if n0 == 0 or n1 == 0:
            class_weight = None
        else:
            # Balance the classes so that the positive class is more important
            # as instructed in the notebook.
            w1 = 2.0 * (n0 / n)
            w0 = (n1 / n)
            class_weight = {1: w1, 0: w0}

        params = {
            "class_weight": class_weight,
        }
        params.update(self._model_params)

        self._model = LogisticRegression(**params)
        self._model.fit(features, y)
        return self
        


    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded.")
        if hasattr(self._model, "predict_proba"):
            p1 = self._model.predict_proba(features)[:, 1]
            return [int(p >= self._decision_threshold) for p in p1]
        return [int(x) for x in self._model.predict(features)]