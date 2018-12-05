from setuptools import setup, find_packages
import os

# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='leap utils',
      version='0.1',
      description='leap_utils',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/janclemenslab/leap_utils',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['numpy', 'videoreader', 'scikit-image', 'keras', 'tensorflow', 'scikit-learn', 'opencv-python', 'matplotlib'],
      extras_require={'train': 'leap'},
      # dependency_links are deprecated so this may stop working in the future
      dependency_links=['git+http://github.com/postpop/videoreader#egg=videoreader-0.2',
                        'git+http://github.com/talmo/leap#egg=leap-0.1'],
      tests_require=['nose'],
      test_suite='nose.collector',
      include_package_data=True,
      zip_safe=False
      )
