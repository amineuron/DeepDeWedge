from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ddw',
    # NOTE (local patch): upstream declares setup_requires=['setuptools_scm'], but the
    # package never uses use_scm_version, so it does nothing here. With setuptools_scm
    # 10.x (split into the separate vcs_versioning package) the build-time fetch crashes
    # with "No module named 'vcs_versioning'". Dropped to make `pip install .` work.
    version='0.0.0',
    python_requires='>=3.7.0',
    author='Simon Wiedemann',
    author_email='simonw.wiedemann@tum.de',
    description='Simultaneous denoising and missing wedge reconstruction of cryo-ET tomograms.',
    packages=['ddw', 'ddw.utils'],
    entry_points={
        'console_scripts': [
            'ddw = ddw.app:main',
        ]},
)