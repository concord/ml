from setuptools import setup

setup(
    name='bfd',
    version='0.1',
    description='ML w/ Concord',
    url='https://github.com/adi-labs/bfd',
    author='Andrew Aday, Alan Du, Carlos Martin, Dennis Wei',
    author_email='alanhdu@gmail.com',
    license='Apache',
    packages=['concord_ml'],
    include_package_data=True,
    install_requires=[
        "concord-py",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ],
    classifiers=['Development Status :: 3 - Alpha'],
    zip_safe=False)
