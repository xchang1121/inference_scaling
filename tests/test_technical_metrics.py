from inference_scaling.retrieval.technical_metrics import numeric_values,technical_proxy


def test_numeric_recall_normalizes_v_but_does_not_accept_partial_or_longer_versions():
    assert numeric_values('Oracle V11.2.0.1 (including RAC)')==['11.2.0.1']
    assert technical_proxy('V11.2.0.1','11.1.0.7 or 11.2',.04)['numeric_recall']==0
    assert technical_proxy('9.0','9.0.1',.4)['numeric_recall']==0
    assert technical_proxy('9.0','Use v9.0.',.4)['numeric_recall']==1


def test_proxy_keeps_no_numeric_cases_and_declares_presence_only():
    assert technical_proxy('table does not exist','table error',.3)['quality_proxy']==.3
    assert technical_proxy('11.2.0.1','11.2.0.1 is unsupported',.2)['numeric_recall']==1
