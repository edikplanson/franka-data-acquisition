from setuptools import setup

package_name = 'camera_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='Camera node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = camera_pkg.camera_node:main',
        ],
    },
)