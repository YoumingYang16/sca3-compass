"""Only reviewed nonstatistical dispatch metadata can differ in parity checks."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_checkpoint_parity import substantive_protocol


def test_known_execution_metadata_ignored_but_science_not_ignored():
    protocol={'seed':123,'repetitions':100,'cases':[{'effect':3.5}],'patterns':True}
    assert substantive_protocol(protocol)==substantive_protocol({**protocol,
        'execution_dispatch':'bounded_worker_count_backlog_v1','continuous_patterns':False})
    for change in [{'seed':124},{'repetitions':101},{'continuous_patterns':True},
                   {'cases':[{'effect':4.5}]},{'unrecognized_setting':False}]:
        assert substantive_protocol(protocol)!=substantive_protocol({**protocol,**change})


def test_unknown_dispatch_rejected():
    with pytest.raises(ValueError,match='Unreviewed'):
        substantive_protocol({'execution_dispatch':'select_favorable_seeds'})
