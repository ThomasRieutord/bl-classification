import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blcovid",
    version="0.3.0",
    author="Meteo-France",
    author_email="thomas.rieutord@meteo.fr",
    description="Boundary layer classification made during Covid-19 outbreak",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasRieutord/bl-classification",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ),
)
