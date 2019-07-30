import setuptools

with open('README.md', 'r') as handle:
    long_description = handle.read()


setuptools.setup(
    name='ensembler',
    version='0.1',
    author='Subigya Jyoti Panta',
    author_email='subigya@t2pco.com',
    description='A package to build ensemble for quick experiments.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="",
    packages=setuptools.find_packages(exclude=['tests*']),
    classifiers=[
        'Programming Language :: Python 3',
        'Licence :: All rights reserved',
        'Operating System :: OS Independent'
    ],
    install_requires=['numpy', 'pandas', 'scikit-learn', 'xgboost']
)