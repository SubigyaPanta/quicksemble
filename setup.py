import setuptools

with open('README.md', 'r') as handle:
    long_description = handle.read()


setuptools.setup(
    name='quicksemble',
    version='0.1',
    author='Subigya Jyoti Panta',
    author_email='subigyapanta@gmail.com',
    description='A package to build ensemble for quick experiments.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/SubigyaPanta/quicksemble",
    packages=setuptools.find_packages(exclude=['tests*']),
    classifiers=[
        'Programming Language :: Python 3',
        'Licence :: All rights reserved',
        'Operating System :: OS Independent'
    ],
    install_requires=['numpy', 'pandas', 'scikit-learn', 'xgboost']
)