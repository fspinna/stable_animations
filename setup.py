from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name='stable_animations',
    version='',
    packages=['stable_animations'],
    url='',
    license='',
    author='Francesco Spinnato',
    author_email='',
    description='',
    install_requires=requirements,
    extras_require={"cuda": ["--extra-index-url https://download.pytorch.org/whl/cu116"]}
)
