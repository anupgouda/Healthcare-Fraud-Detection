import pandas as pd
import shap


class FraudExplainer:

    def __init__(self, model, features):

        self.model = model
        self.features = features

        self.explainer = shap.TreeExplainer(
            self.model
        )

    def explain(self, provider_data):

        # ------------------------------------------------
        # CONVERT SINGLE PROVIDER SERIES TO DATAFRAME
        # ------------------------------------------------

        if isinstance(
            provider_data,
            pd.Series
        ):

            provider_data = provider_data.to_frame().T


        # ------------------------------------------------
        # CONVERT DICT TO DATAFRAME
        # ------------------------------------------------

        elif isinstance(
            provider_data,
            dict
        ):

            provider_data = pd.DataFrame(
                [provider_data]
            )


        # ------------------------------------------------
        # SELECT FEATURES
        # ------------------------------------------------

        provider_data = provider_data[
            self.features
        ]


        # ------------------------------------------------
        # CALCULATE SHAP VALUES
        # ------------------------------------------------

        shap_values = self.explainer.shap_values(
            provider_data
        )


        # ------------------------------------------------
        # HANDLE SHAP OUTPUT
        # ------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            values = shap_values[1][0]

        else:

            values = shap_values[0]

            if len(values.shape) > 1:

                values = values[:, 1]


        # ------------------------------------------------
        # CREATE EXPLANATION
        # ------------------------------------------------

        explanation = pd.DataFrame(
            {
                "Feature": self.features,

                "SHAP Value": values,

                "Feature Value": (
                    provider_data.iloc[0].to_numpy()
                )
            }
        )


        # ------------------------------------------------
        # CALCULATE IMPACT
        # ------------------------------------------------

        explanation["Impact"] = (
            explanation["SHAP Value"]
            .abs()
        )


        # ------------------------------------------------
        # SORT BY IMPORTANCE
        # ------------------------------------------------

        explanation = explanation.sort_values(
            "Impact",
            ascending=False
        ).reset_index(
            drop=True
        )


        return explanation