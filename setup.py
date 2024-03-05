from setuptools import setup

setup(
    name='TibetanOCR',
    url='https://github.com/eric86y/TibetanOCR',
    author='Eric Werner',
    author_email='eric.werner.mail@gmail.com',
    packages=['TibetanOCR'],
    install_requires=[
        "fast-ctc-decode=>0.3.5",
        "numpy>=1.26",
        "onnxruntime>=1.17",
        "opencv-python>=4.9",
        "pyewts",
        "tqdm=>4.66"  
    ],
    version='0.1',
    license='MIT',
    description='A package for running OCR Models for Tibetan',
    long_description=open('README.md').read(),
)