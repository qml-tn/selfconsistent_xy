
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='selfconsistent_xy',
    version='0.0.1',
    description='From classical to quantum machine learning with tensor networks',
    long_description=readme,
    author='Bojan Žunkovič',
    author_email='bojan.zunkovic@fri.uni-lj.si',
    url='https://github.com/znajob/qml-tn.git',
    license=license,
    packages=["selfconsistent_xy"],
)
