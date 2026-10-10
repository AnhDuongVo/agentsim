# Validation evidence — 9 October 2026

`pip install -e ".[dev]"` succeeded in a fresh virtual environment for this repository, independently of the other projects. Tests ran on Python 3.12.14 / Darwin arm64 CPU. Other Python versions in CI have not been executed locally here.

| Check | Result |
|---|---|
| Offline test suite | 18 passed |
| Ruff lint | Passed |
| Ruff formatting | Passed |
| Mocked/simulated integration | Tested within the scope below |
| Live NVIDIA hosted endpoint | Not executed |
| Self-hosted GPU endpoint | Not executed |
| Clinical/scientific domain validation | Not completed |

## Tested scope

Simulation, streaming server usage versus chunks/estimates, request failures, exclusion of estimated token counts from calibration, estimated prompt-length exclusion from prefill fit.

The GitHub workflows have been added or retained, but their remote execution has not been verified after these changes. Unit tests establish behavior on fixtures; they do not establish semantic or clinical correctness.

Calibration excludes explicitly estimated completion counts. When prompt counts are estimated, the prefill fit retains its baseline rate rather than inferring a measured rate from approximate prompt lengths. Existing example measurements with no provenance are legacy illustrative inputs and should not be presented as measured hardware evidence.
