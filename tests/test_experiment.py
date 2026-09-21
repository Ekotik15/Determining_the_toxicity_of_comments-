from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from toxic_comments.experiment import _linear_svc_pipeline


def test_linear_svc_ablation_changes_only_features() -> None:
    word_only = _linear_svc_pipeline(5000, random_state=42, include_char_features=False)
    word_char = _linear_svc_pipeline(5000, random_state=42, include_char_features=True)

    assert isinstance(word_only["features"], TfidfVectorizer)
    assert isinstance(word_char["features"], FeatureUnion)

    word_only_classifier = word_only["classifier"]
    word_char_classifier = word_char["classifier"]
    assert isinstance(word_only_classifier, CalibratedClassifierCV)
    assert isinstance(word_char_classifier, CalibratedClassifierCV)
    for parameter in (
        "cv",
        "method",
        "estimator__C",
        "estimator__class_weight",
        "estimator__random_state",
    ):
        assert (
            word_only_classifier.get_params()[parameter]
            == word_char_classifier.get_params()[parameter]
        )
