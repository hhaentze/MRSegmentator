# tests/conftest.py


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "body_comp: mark test as requiring the body-composition model",
    )
