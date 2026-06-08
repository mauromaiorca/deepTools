from setuptools import setup, find_packages

setup(
    name="deepTools",
    version="0.1.0",
    description="A package for cryo-EM processing.",
    author="Mauro",
    author_email="your.email@example.com",
    url="https://github.com/mauromaiorca/deepTools", 
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'deepTools': ['infer_config.json', 'environment.yml']
    },
    entry_points={
        'console_scripts': [
            'deepTools=deepTools.infer:main',
            'deepTools_train=deepTools.train:main',
            'deepTools_map_process=deepTools.map_processing:main',
            'deepTools_model_process=deepTools.model_processing:main'
        ]
    },
    install_requires=[
        "numpy>=1.21.6,<1.28.0",
        "mrcfile",
        "scipy>=1.8.0,<1.11.0",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "torchmetrics",
        "flatbuffers>=1.12",
        "gast>=0.2.1",
        "keras>=2.8.0rc0,<2.9",
        "wrapt>=1.11.0",
        "tensorboard>=2.8.0,<2.9",
        "SimpleITK",
        "scikit-image",
        "pandas>=1.3.0,<2.0.0",
        "starfile>=0.0.8,<1.0.0",
        "cupy-cuda117>=10.2.0,<11.0.0",
        "tqdm>=4.0.0,<5.0.0",
        "biopython==1.73",
        "pyyaml"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

