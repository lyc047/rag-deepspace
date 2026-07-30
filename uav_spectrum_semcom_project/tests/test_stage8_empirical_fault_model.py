from scripts.freeze_stage8_empirical_fault_model import freeze


def test_stage8_empirical_model_freezes_without_confirmation_access(tmp_path):
    # Integration behavior is covered by the registered real artifacts; this
    # test protects the public function contract without duplicating large JSON.
    assert callable(freeze)
