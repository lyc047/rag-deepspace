import copy
import json
from pathlib import Path

import pytest

from scripts.register_stage8_readiness import (
    ROOT,
    register,
    validate_external,
)


def test_stage8_readiness_registration_is_nonexecuting(tmp_path: Path) -> None:
    result = register(
        ROOT / "configs/stage8_external_final_readiness_v1.json",
        ROOT / "configs/stage8_hardware_readiness_v1.json",
        tmp_path / "result.json",
    )
    assert result["status"] == "REGISTERED_NOT_EXECUTABLE"
    assert result["decision"]["candidate_integrity_passed"] is True
    assert result["decision"]["execution_authorized"] is False
    assert result["external_final"]["minimum_new_independent_units"] >= 20
    assert result["external_final"]["signal_access_allowed_now"] is False


def test_stage8_external_protocol_rejects_post_access_retuning() -> None:
    path = ROOT / "configs/stage8_external_final_readiness_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    invalid = copy.deepcopy(config)
    invalid["access_governance"]["post_access_parameter_retuning"] = True
    with pytest.raises(ValueError, match="retuning"):
        validate_external(invalid)


def test_stage8_external_protocol_binds_candidate_hash() -> None:
    path = ROOT / "configs/stage8_external_final_readiness_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    invalid = copy.deepcopy(config)
    invalid["candidate_freeze"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_external(invalid)
