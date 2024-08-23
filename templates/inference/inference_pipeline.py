"""Inference pipeline for standard AI use cases"""

from inference_runner import InferenceRunner, ConvNetRunner, SingleClassificationRunner, MultiClassificationRunner, RegressionRunner

MODEL_REGISTRY = {
    "conv-net": ConvNetRunner(),
    "classification-single": SingleClassificationRunner(),
    "classification-multi": MultiClassificationRunner(),
    "regression": RegressionRunner(),
}

def pipeline(model_id: str) -> InferenceRunner:
    runner = MODEL_REGISTRY.get(model_id)
    if runner is None:
        raise ValueError(f"Provided model_id `{model_id}` does not exist.")
    return runner
