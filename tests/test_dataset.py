
from src.data.manifest import VIEWS


def test_required_views_defined():
    assert VIEWS == ("anterior", "oblleft", "oblright")
