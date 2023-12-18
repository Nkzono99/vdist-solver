from setuptools import find_packages, setup


def _require_packages(filename):
    return open(filename).read().splitlines()


long_description = open('README.md', 'r', encoding='utf-8').read()

setup(
    name='vdist-solver',
    description='Velocity distribution function solver using Liouville\'s theorem',
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='1.0.3',
    install_requires=_require_packages('requirements.txt'),
    author='Nkzono99',
    author_email='210x218x@gsuite.stu.kobe-u.ac.jp',
    url='https://github.com/Nkzono99/vdist-solver',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'gen-vdsolver = vdsolver.tools.gentemp:gentemp_vdsolver',
            'gen-backtrace = vdsolver.tools.gentemp:gentemp_backtrace',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
