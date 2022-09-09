import re

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


def get_version():
    with open("src/sprite/version.py", "r") as version_file:
        return re.search(r'__version__\s*=\s*"(.*)"', version_file.read()).group(1)


ext_modules = [
    Pybind11Extension(
        "sprite/spelling_correction/models/mlp/bktree",
        [r"src/sprite/spelling_correction/models/mlp/bktree.cpp"],
        define_macros=[("VERSION_INFO", get_version())],
    ),
    Pybind11Extension(
        "sprite/spelling_correction/models/mlp/counting_utils",
        [r"src/sprite/spelling_correction/models/mlp/counting_utils.cpp"],
        define_macros=[("VERSION_INFO", get_version())],
    ),
]

setup(
    name="sprite",
    version=get_version(),
    description="nlp processsing",
    keywords="nlp pipeline",
    url="https://gitlab.jingle.cn/dataplatform/ml/nlp/sprite",
    python_requires=">=3.7",
    packages=find_packages("src", exclude=["*test*", "*example*"]),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=[
        "numpy>=1.20.3",
        "pybind11>=2.6.1,<=3.0.0",
        "hdfs==2.6.0",
        "emoji>=1.4.2,<=2.0.0",
        "nltk==3.5",
        "wordninja==2.0.0",
        "python-Levenshtein==0.12.0",
        "torch>=1.8.2,<=2.0.0",
        "metaphone==0.6.0",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
