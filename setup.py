from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="modelrunner",
    version="0.1",
    packages=find_packages(),
    author="Hamid Shojaei",
    author_email="hamidreza.shojaei@gmail.com",
    description="A python library that apply multiple predictive models on a dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hamid-shojaei/modelrunner",
    install_requires=[
        "pandas",
        "numpy",
        ...
    ],
    extras_require={
        'svg': ["cairosvg"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)