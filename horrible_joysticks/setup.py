from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'horrible_joysticks'

config_files = [p for p in glob(os.path.join("config", "**", "*"), recursive=True)
                if os.path.isfile(p)]
launch_files = [p for p in glob(os.path.join("launch", "**", "*"), recursive=True)
                if os.path.isfile(p)]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), config_files),
        (os.path.join('share', package_name, 'launch'), launch_files),
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
            f"microphone_power_spectrum = {package_name}.nodes.microphone_power_spectrum:main",
            f"spectrogram_gui = {package_name}.nodes.spectrogram_gui:main",
            f"harmonica_joystick = {package_name}.nodes.harmonica_joystick:main",
        ],
    },
)
