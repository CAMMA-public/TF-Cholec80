from setuptools import setup, find_packages

setup(
  name="tf_cholec80",
  version="0.1.0",
  description="Cholec80 for Tensorflow",
  author="Tong Yu",
  author_email="yu.tong.nicolas@gmail.com",
  py_modules=["tf_cholec80"],
  packages=find_packages(include=["tf_cholec80"]),
  install_requires=[
    "tensorflow>=1.4.0"
  ],
  package_data={"tf_cholec80": ["configs/config.json"]}
)