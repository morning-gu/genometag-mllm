"""脚手架验证：包与全部子包可导入。"""


def test_package_importable():
    import genometag
    assert genometag is not None


def test_subpackages_importable():
    import genometag.data  # noqa: F401
    import genometag.tags  # noqa: F401
    import genometag.annotate  # noqa: F401
    import genometag.eval  # noqa: F401
    import genometag.baselines  # noqa: F401
