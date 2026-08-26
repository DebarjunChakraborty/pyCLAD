import pytest

from pyclad.data.datasets.opssat_dataset import OpsSatDataset


def test_opssat_dataset_loading():
    dataset = OpsSatDataset(channel="CADC0874")
    assert dataset.name() == "OPS-SAT-CADC0874"
    assert len(dataset.train_concepts()) == 96
    assert len(dataset.test_concepts()) == 1

    first_concept = dataset.train_concepts()[0]
    assert first_concept.data.shape == (1, 18)
    assert first_concept.name == "CADC0874_train_0000"


def test_opssat_invalid_channel():
    with pytest.raises(Exception):
        OpsSatDataset(channel="INVALID_CHANNEL_NAME")
