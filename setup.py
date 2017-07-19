"""Tensorflow experiments for use with google cloud."""
import setuptools

_PANTS_REQUIREMENTS_PATH="3rdparty/python/requirements.txt"

def get_pants_requirements():
    """Parse the pants requirements file."""
    f = open(_PANTS_REQUIREMENTS_PATH, 'r')
    return [line.strip() in f.readlines()]

setuptools.setup(
    name="tensorflow_experiments",
    version="0.0.1",
    include_package_data=True,
    packages=setuptools.find_packages(),
    author="Leopold Haller",
    author_email="leopoldhaller@gmail.com")
