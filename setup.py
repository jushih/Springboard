from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='fashion',
    version='0.1',
    description='Analysis of the DeepFashion dataset',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/jushih/Springboard',
    author='Julie Shih',
    packages=['fashion'],
    install_requires=[
        'pypandoc>=1.4',
        'matplotlib>=3.0.3',
        'numpy>=1.16.3',
        'pandas>=0.24.2'
        ]
)

