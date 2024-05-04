from setuptools import find_packages, setup

package_name = 'omnibot_nav'

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
    maintainer='xianzhou',
    maintainer_email='1544453976@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'envmodel = omnibot_nav.EnvModel:main',
            'omnibot_nav = omnibot_nav.main:main'
        ],
    },
)
