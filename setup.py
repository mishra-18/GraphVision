from setuptools import setup, find_packages

setup(
    name='graphvision',
    version='0.1.3',
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.5.1',
        'numpy>=1.22.0',
        'torch>=1.2.0',
        'transformers>=4.36.2',
        'networkx>=2.7',
        'torch_geometric>=2.0.4',
        'opencv-python>=4.5',
        'pillow>=9.0.1',
        'huggingface-hub',
        'segment-anything',
    ],
    author='mishra-18',
    author_email='mishra4475@gmail.com',
    description='Create topological graph for image segments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
