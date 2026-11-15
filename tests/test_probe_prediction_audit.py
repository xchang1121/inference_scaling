"""Numerical tolerance must never change gating semantics or hide larger drift."""
import math
from audit_probe_online_is import prediction_matches


def test_native_one_ulp_and_exact_replay():
    assert prediction_matches(.6879629618447612,.6879629618447614,.8)
    assert prediction_matches(.6879629618447612,.6879629618447614,.9)
    assert prediction_matches(.5,.5,.5)


def test_rejects_even_one_ulp_threshold_crossing():
    lo=math.nextafter(.5,0.)
    hi=math.nextafter(.5,1.)
    assert not prediction_matches(lo,.5,.5)
    assert not prediction_matches(hi,.5,.5,risk=True)


def test_rejects_nonfinite_and_larger_drift():
    assert not prediction_matches(.5,.5+3*math.ulp(.5),.9)
    for value in (float('nan'),float('inf'),-float('inf')):
        assert not prediction_matches(value,.5,.9)
        assert not prediction_matches(.5,value,.9)
