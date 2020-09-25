from setuptools import setup, find_packages

setup(
  name="tf_cholec80",
  version="0.1.0",
  description="Cholec80 for Tensorflow",
  author="Tong Yu",
  author_email="yu.tong.nicolas@gmail.com",
  packages=find_packages(include=["tf_cholec80"]),
  package_data={"tf_cholec80": ["configs/config.json"]}
)