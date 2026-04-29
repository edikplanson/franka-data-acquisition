from setuptools import setup

setup(
    name='my_robot',
    version='0.1.0',
    packages=['my_robot'],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'tactile_node = my_robot.tactile_node:main',
            # 'camera_node = my_robot.camera_node:main',  ← décommenter quand prêt
        ],
    },
)