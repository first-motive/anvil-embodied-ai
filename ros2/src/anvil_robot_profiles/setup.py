from glob import glob

from setuptools import find_packages, setup

package_name = "anvil_robot_profiles"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/profiles", glob("profiles/*.yaml")),
    ],
    python_requires=">=3.12",
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nishalan Govender",
    maintainer_email="nish@ubundi.co.za",
    description="Robot profiles: joint, controller and frame names per robot build",
    license="Apache-2.0",
    tests_require=["pytest"],
)
