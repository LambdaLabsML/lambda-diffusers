from setuptools import setup, find_packages

setup(
    name='lambda-diffusers',
    version='0.0.1',
    description='Lambda Labs additions to Diffusers',
    author='Justin Pinkney',
    author_email='justin@lambdal.com',
    url='https://github.com/LambdaLabsML/lambda-diffusers',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'Pillow',
        'transformers',
        'diffusers',
    ],
)