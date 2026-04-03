from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'auv_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(
            os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nikhil',
    maintainer_email='dev@todo.todo',
    description='AUV Vision Pipeline with OpenCL CLAHE',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = auv_vision.camera_node:main',
            'clahe_node = auv_vision.clahe_node:main'
        ],
    },
)
