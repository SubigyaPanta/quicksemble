import setuptools

with open('README.md', 'r') as handle:
    long_description = handle.read()


setuptools.setup(
    name='quicksemble',
    version='0.2.1',
    author='Subigya Jyoti Panta',
    author_email='subigyapanta@gmail.com',
    description='A package to build ensemble for quick experiments.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/SubigyaPanta/quicksemble",
    packages=setuptools.find_packages(exclude=['tests*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent'
    ],
    install_requires=['numpy', 'scikit-learn', 'xgboost'],
    project_urls={
            'Bug Reports': 'https://github.com/SubigyaPanta/quicksemble/issues',
            'Source': 'https://github.com/SubigyaPanta/quicksemble',
        }
)