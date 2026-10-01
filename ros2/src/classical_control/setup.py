from glob import glob

from setuptools import find_packages, setup

package_name = "classical_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    python_requires=">=3.12",
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nishalan Govender",
    maintainer_email="nish@ubundi.co.za",
    description="Classical look-then-move can-on-paper baseline for the OpenArm right arm",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
