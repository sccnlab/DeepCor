from setuptools import setup, find_packages

setup(
    name='DeepCor',
    version='0.1.0', 
    packages=find_packages(),
    install_requires=[
        "nibabel>=3.2.0",
        "numpy>=1.21.5",
        "matplotlib>=3.5.1",
        "scipy>=1.7.3",
        "scikit-learn>=1.2.2",
        "torch>=1.11.1"
    ],
    author='Yu Zhu',
    description='This is a deep generative method to denoise the fMRI data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sccnlab/DeepCor/DeepCor_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)