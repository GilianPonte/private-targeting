from private_targeting import __version__, CTENN, DP_CATE, DP_policy


def test_version_is_present():
    assert isinstance(__version__, str)
    assert __version__


def test_public_api_exists():
    assert callable(CTENN)
    assert callable(DP_CATE)
    assert callable(DP_policy)
