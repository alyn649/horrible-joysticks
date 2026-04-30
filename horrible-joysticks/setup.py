from setuptools import find_packages, setup

package_name = 'horrible-joysticks'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='angus',
    maintainer_email='alyn649@aucklanduni.ac.nz',
    description='The goal of this package is to create some joysticks available to use in ros, which no one will ever want to use.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
