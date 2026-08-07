"""Cross-implementation verification against a matched pymgrid plant.

This experiment deliberately verifies only the shared, lossless scheduling
contract.  It does not treat pymgrid as physical ground truth and does not
compare either simulator's native reward.  Both implementations receive the
same exogenous series and open-loop actions; their outputs are normalized into
one accounting schema before deltas are computed.

The package is split by responsibility:

* ``constants`` experiment identity and reference provenance
* ``plant``     shared plant parameters, fixture steps, acceptance criteria
* ``harness``   per-implementation runners and canonical row normalization
* ``compare``   field-by-field trajectory comparison and tolerances
* ``report``    human-readable markdown report rendering
* ``runner``    artifact persistence and the CLI entry point
"""

from microgrid_simulator.experiments.pymgrid_verification.compare import (
    COMPARISON_FIELDS,
    SUMMARY_LABELS,
    _compare,
)
from microgrid_simulator.experiments.pymgrid_verification.constants import (
    CANDIDATE_IMPLEMENTATION,
    PYMGRID_UPSTREAM_COMMIT,
    REFERENCE_IMPLEMENTATION,
)
from microgrid_simulator.experiments.pymgrid_verification.harness import (
    _build_pymgrid,
    _canonical_row,
    _info_value,
    _pymgrid_battery_action_mwh,
    _run_pymgrid,
    _run_simple,
    _summarize_rows,
)
from microgrid_simulator.experiments.pymgrid_verification.plant import (
    AcceptanceCriteria,
    CommonPlant,
    FixtureStep,
    _common_settings,
    _NoDegradationBattery,
    common_fixture,
)
from microgrid_simulator.experiments.pymgrid_verification.report import _report
from microgrid_simulator.experiments.pymgrid_verification.runner import main, run_verification

__all__ = [
    "AcceptanceCriteria",
    "CANDIDATE_IMPLEMENTATION",
    "COMPARISON_FIELDS",
    "CommonPlant",
    "FixtureStep",
    "PYMGRID_UPSTREAM_COMMIT",
    "REFERENCE_IMPLEMENTATION",
    "SUMMARY_LABELS",
    "_NoDegradationBattery",
    "_build_pymgrid",
    "_canonical_row",
    "_common_settings",
    "_compare",
    "_info_value",
    "_pymgrid_battery_action_mwh",
    "_report",
    "_run_pymgrid",
    "_run_simple",
    "_summarize_rows",
    "common_fixture",
    "main",
    "run_verification",
]
