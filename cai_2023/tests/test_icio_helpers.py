import numpy as np
import pytest
from gravity_icio.icio import _country_industry


def test_country_industry_parser():
    assert _country_industry(["CAN_01","USA_02"]) == [("CAN","01"),("USA","02")]


def test_country_industry_parser_rejects_bad_label():
    with pytest.raises(ValueError):
        _country_industry(["CAN01"])
