"""Chimera: Hierarchy-Based Multi-Objective Optimization
"""

from setuptools import setup
import versioneer


def readme():
    with open('README.md', 'r') as f:
        return f.read()


# -----
# Setup
# -----
setup(name='matter-chimera',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
      ],
      url='https://github.com/aspuru-guzik-group/chimera',
      author='Florian Häse, Matteo Aldeghi',
      author_email='matteo.aldeghi@vectorinstitute.ai',
      license='MIT',
      packages=['chimera'],
      package_dir={'': 'src'},
      zip_safe=False,
      install_requires=['numpy'],
      python_requires=">=3"
      )
