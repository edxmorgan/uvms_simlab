from setuptools import find_packages, setup
from glob import glob
package_name = 'simlab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/diff_iK.casadi']),
        ('share/' + package_name, ['resource/fk_eval.casadi']),
        (
            'share/' + package_name + '/manipulator',
            glob('resource/manipulator/*')
        ),
        (
            'share/' + package_name + '/vehicle',
            glob('resource/vehicle/*')
        ),
        ('lib/' + package_name, [package_name+'/robot.py']),
        ('lib/' + package_name, [package_name+'/blue_rov.py']),
        ('lib/' + package_name, [package_name+'/alpha_reach.py']),
        ('lib/' + package_name, [package_name+'/task.py']),
        ('lib/' + package_name, [package_name+'/shape_task.py']),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mr-robot',
    maintainer_email='edmorgangh@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joystick_controller = simlab.joystick_control:main',
            'sensor_py_node = simlab.sensor:main',
            'excite_py_node = simlab.excite_effort:main',
            'coverage_node = simlab.dive_coverage:main',
            'shape_formation_node = simlab.shape_formation:main',
            'station_node = simlab.station:main',
            'ik_solve_node = simlab.ik_solve:main',
            'pwm_test_node = simlab.pwm_test:main',
            'pid_metrics_node = simlab.pid_metrics:main'
        ],
    },
)
