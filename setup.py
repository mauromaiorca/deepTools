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
            'deepTools_model_process=deepTools.model_processing:main',
            'deepTools_setup=deepTools.setup_config:main'
        ]
    },
    install_requires=[
        "numpy>=1.21.6,<1.28.0",
        "mrcfile",
        "scipy>=1.8.0,<1.11.0",
        "torch==2.0.0",
        "torchvision==0.15.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

