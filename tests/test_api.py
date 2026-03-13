from private_targeting import __version__, cnn, ctenn, dp_cate, pcnn


def test_version_is_present():
    assert isinstance(__version__, str)
    assert __version__



def test_aliases_exist():
    assert callable(cnn)
    assert callable(pcnn)
    assert callable(ctenn)
    assert callable(dp_cate)
