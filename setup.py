from typing import List
from setuptools import find_packages, setup


def get_packages(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [x.replace('\n', "") for x in requirements ]
        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements



setup(
    name= "IDS",
    version = "0.0.1",
    author = "Mohammed Saim Ahmed Quadri",
    author_email="mohammedsaimquadri@gmail.com",
    packages= find_packages(),
    install_requires = get_packages("requirements.txt")
)