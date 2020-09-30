import nox


@nox.session()
def lint(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements/dev.txt")
    session.run("flake8", "activemri")
    # session.run("black", "--check", "activemri")


@nox.session()
def mypy(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements/dev.txt")
    session.run("mypy", "activemri")


@nox.session()
def pytest(session) -> None:
    session.install("--upgrade", "setuptools", "pip")
    session.install("pyxb==1.2.6")
    session.install("torch")
    session.install("torchvision")
    session.install("-e", ".")
    session.run("pytest", "tests/core")
