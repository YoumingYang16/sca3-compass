import numpy as np
import pytest

from sca3_compass.molecular_block_validation import unit_partition


def test_every_region_of_held_animals_is_excluded():
    samples=[{"unit_id":str(i),"tissue":r} for i in range(14) for r in ("brainstem","cortex","cerebellum","striatum")]
    train,test=unit_partition(samples,("2","11"))
    assert len(train)==48 and len(test)==8
    assert {samples[i]["unit_id"] for i in test}=={"2","11"}
    assert not {"2","11"}&{samples[i]["unit_id"] for i in train}


def test_unresolved_identity_cannot_be_randomly_split():
    with pytest.raises(ValueError):
        unit_partition([{"unit_id":None}],("1","2"))


def test_partition_does_not_depend_on_any_expression_values():
    samples=[{"unit_id":str(i)} for i in range(8)]
    x=np.random.default_rng(1).normal(size=(30,8))
    before=unit_partition(samples,("1","2"))
    x[:,[1,2]]=1e9
    assert unit_partition(samples,("1","2"))==before
