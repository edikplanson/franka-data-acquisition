from setuptools import setup

package_name = 'dataset_logger'

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
    description='Dataset logger',
    license='MIT',
    entry_points={
        'console_scripts': [
            'dataset_logger = dataset_logger.logger_node:main',
        ],
    },
)