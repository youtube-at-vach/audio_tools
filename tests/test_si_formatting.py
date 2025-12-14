from src.core.utils import format_si


def test_format_si_zero():
    assert format_si(0.0, "S") == "0 S"


def test_format_si_invalid():
    assert format_si(float("nan"), "S") == "-"
    assert format_si(float("inf"), "S") == "-"


def test_format_si_admittance_prefixes():
    assert format_si(0.00123, "S", sig_figs=4) == "1.23 mS"
    assert format_si(1.234e-6, "S", sig_figs=4) == "1.234 µS"
    assert format_si(-2.5e-9, "S", sig_figs=3) == "-2.5 nS"


def test_format_si_rounding_spillover():
    # 0.9996 S is 999.6 mS; with 3 sig figs it should display as 1 S (spillover fix).
    assert format_si(0.9996, "S", sig_figs=3) == "1 S"
