import numpy as np
import predictive_bridge as pb


def test_rank_plus_one_and_ties():
    refs = np.array([-1., 0., 1., 2.])
    np.testing.assert_array_equal(pb.rank_tail(np.array([-1., 1., 2., 3.]), refs), [1., .6, .4, .2])


def test_fourfold_pilot_provenance():
    rng = np.random.default_rng(918)
    z = rng.normal(size=(32, 4, 6)); cal = rng.normal(size=(4, 4, 6))
    a = pb.evaluate(z, cal, seed=51, reference_draws=31)
    assert a['status'] == 'completed'
    z[0] += 7  # HELD for fold0; cannot enter fold0 pilot or learner
    b = pb.evaluate(z, cal, seed=51, reference_draws=31)
    for key in ['gamma_from_pilot', 'profiles', 'shape']:
        np.testing.assert_array_equal(a['folds'][0][key], b['folds'][0][key])
    assert a['folds'][0]['pilots'] == b['folds'][0]['pilots']
    # External calibration may change learned directions but not pilot choice.
    c = pb.evaluate(z, cal+2, seed=51, reference_draws=31)
    assert b['folds'][0]['pilots'] == c['folds'][0]['pilots']
    np.testing.assert_array_equal(b['folds'][0]['gamma_from_pilot'], c['folds'][0]['gamma_from_pilot'])


def test_failure_is_recorded_no_discovery(monkeypatch):
    def fail(*args, **kwargs): raise ArithmeticError('injected numerical failure')
    monkeypatch.setattr(pb, 'shape_fit', fail)
    z = np.random.default_rng(11).normal(size=(32, 4, 6))
    r = pb.evaluate(z, z[:4], seed=1, reference_draws=15)
    assert r['status'] == 'conservative_numerical_failure'
    assert not r['decisions']['PB_main'].any()


def test_external_bound_monotone_and_pilot_unchanged():
    rng=np.random.default_rng(921)
    z=rng.normal(size=(32,4,6))+1; cal=rng.normal(size=(8,4,6))
    a=pb.evaluate(z,cal,seed=21,reference_draws=63)
    b=pb.evaluate(z,cal,seed=21,reference_draws=63,mismatch_bound=5)
    assert a['status']==b['status']=='completed'
    assert np.all(b['p']>=a['p'])
    assert np.all(b['evidence']['PB_grid']<=a['evidence']['PB_grid']+1e-12)
    for x,y in zip(a['folds'],b['folds']):
        assert x['pilots']==y['pilots']
        np.testing.assert_array_equal(x['profiles'],y['profiles'])
