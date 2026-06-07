from setuptools import setup, find_packages

setup(
    name="dexpbt_isaaclab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
    package_data={
        "dexpbt_isaaclab": ["agents/*.yaml"],
    },
    python_requires=">=3.10",
)
