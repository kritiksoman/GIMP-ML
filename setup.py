from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="gimpml",  # Required
    version="0.0.7",  # Required
    description="A.I. for GIMP",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/kritiksoman/GIMP-ML",  # Optional
    author="Kritik Soman",  # Optional
    author_email="kritiksoman@ieee.org",  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.8",
        # 'Programming Language :: Python :: 2.7 :: Only',
    ],
    keywords="sample, setuptools, development",  # Optional
    packages=find_packages(),
    python_requires=">=2.7",
    include_package_data=True,  # to incluse manifest.in
    install_requires=[
        "numpy",
        'future; python_version <= "2.7"',
        "scipy",
        "gdown",
        "typing",
        'enum; python_version <= "2.7"',
        "requests",
        "opencv-python<=4.3",
        "pretrainedmodels",
        "scikit-image",
        "timm==0.4.5",
    ]  # , "torch==1.8", "torchvision"],
)
