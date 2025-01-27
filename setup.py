from setuptools import find_packages, setup

package_name = 'simlab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/lookup_function_0.casadi']),
        ('share/' + package_name, ['resource/lookup_function_1.casadi']),
        ('share/' + package_name, ['resource/lookup_function_2.casadi']),
        ('share/' + package_name, ['resource/lookup_function_3.casadi']),
        ('share/' + package_name, ['resource/ref_intg.casadi']),
        ('share/' + package_name, ['resource/J_uvms.casadi']),
        ('lib/' + package_name, [package_name+'/robot.py']),
        ('lib/' + package_name, [package_name+'/task.py']),
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
            'mouse_node_effort = simlab.mouse_node_effort:main',
            'sensor_py_node = simlab.sensor:main',
            'excite_py_node = simlab.excite_effort:main',
            'coverage_node = simlab.dive_coverage:main'
        ],
    },
)
