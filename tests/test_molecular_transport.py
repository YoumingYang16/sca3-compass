import itertools

import numpy as np
import pytest

from sca3_compass.molecular_transport import (
    contrast_statistics,
    exact_assignments,
    holm,
    normalized_views,
    permutation_contrasts,
)


def test_exact_assignments_count_balance_unique():
    labels=exact_assignments(12,6)
    assert labels.shape==(924,12)
    assert (labels.sum(axis=1)==6).all()
    assert np.unique(labels,axis=0).shape==labels.shape


def test_vectorized_permutations_match_scalar_reference():
    x=np.arange(18).reshape(3,6).astype(float)**1.3
    labels=exact_assignments(6,3)
    result=permutation_contrasts(x,labels)
    expected=np.array([x[:,l].mean(axis=1)-x[:,~l].mean(axis=1) for l in labels])
    np.testing.assert_allclose(result,expected,atol=1e-12)


def test_exact_null_rank_superuniform_including_ties():
    x=np.array([[1,1,3,5,5,7]],float)
    contrasts=permutation_contrasts(x,exact_assignments(6,3))[:,0]
    p=np.array([np.mean(contrasts>=c-1e-12) for c in contrasts])
    for threshold in [.05,.1,.2,.5,.9,1]:
        assert np.mean(p<=threshold)<=threshold+1e-12


def test_holm_matches_exhaustive_closed_bonferroni():
    p=np.array([.001,.03,.02,.8])
    expected=np.zeros(4)
    for size in range(1,5):
        for subset in itertools.combinations(range(4),size):
            global_p=min(1,size*min(p[list(subset)]))
            for i in subset:
                expected[i]=max(expected[i],global_p)
    np.testing.assert_allclose(holm(p),expected)


def test_development_normalizations_label_invariant_and_finite():
    x=np.array([[0,2,1,4],[2,3,4,5]],float)
    views=normalized_views(x)
    assert len(views)==3
    for view in views:
        assert np.isfinite(view).all()
        effect,t=contrast_statistics(view,np.array([False,False,True,True]))
        assert effect.shape==t.shape==(2,)


@pytest.mark.parametrize("n,k",[(2,0),(2,2),(20,10)])
def test_unsafe_enumerations_refused(n,k):
    with pytest.raises(ValueError):
        exact_assignments(n,k)
