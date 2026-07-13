"""Smoke test: verify package structure and creator dependency are accessible."""

from elasticai.creator.experimental.ir.implementation_provider.implementation_provider import (
    TrainingImplementationProvider,
    apply_training_provider,
)


def test_odt_package_importable():
    from odt import providers, ir2c, resource_estimator

    assert providers is not None
    assert ir2c is not None
    assert resource_estimator is not None


def test_creator_protocol_accessible():
    assert hasattr(TrainingImplementationProvider, "model_attributes")
    assert hasattr(TrainingImplementationProvider, "training_function")
    assert hasattr(TrainingImplementationProvider, "optimizer")
    assert hasattr(TrainingImplementationProvider, "loss")


def test_apply_training_provider_callable():
    assert callable(apply_training_provider)
