from sklearn.metrics import r2_score
from typing import Optional


def get_r2_score(
    y_train, y_hat, num_places=None, text=False
) -> Tuple[Optional[str, float]]:
    """Get accuracy score for model results."""

    score = r2_score(y_train, y_hat)

    if num_places:
        score = round(score, num_places)
        msg = f"\tAccuracy score (.{num_places}f) : \n\t\tScore : [ {score} ]"
    else:
        l = len(str(score))
        msg = f"\tAccuracy score (.{l}f) : \n\t\tScore :  : [ {score} ]"

    return msg if text else score
