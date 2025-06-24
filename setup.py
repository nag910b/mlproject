from math import hypot
from setuptools import setup, find_packages
from typing import List
HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] #each and every time for next line /n is recorded so we have to replace it with empty string
        if HYPEN_E_DOT in requirements:   #-e . is used to install the package in the current directory. If we dont want to install the package in the current directory then we can remove it. It will not come if we make that requirement.txt manually(int this case).
            #but we are doing it in general so we have to remove it as a good practice.
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name='mlproject',
    version='0.0.1',
    author='Satya',
    author_email='sainag910@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),


    )