# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup


VERSION = "0.9.1.dev0"

extras = {}
extras["quality"] = [
    "black",  # doc-builder has an implicit dependency on Black, see huggingface/doc-builder#434
    "hf-doc-builder",
    "ruff~=0.2.1",
]
extras["docs_specific"] = [
    "black",  # doc-builder has an implicit dependency on Black, see huggingface/doc-builder#434
    "hf-doc-builder",
]
extras["dev"] = extras["quality"] + extras["docs_specific"]
extras["test"] = extras["dev"] + [
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "parameterized",
    "datasets",
    "diffusers<0.21.0",
    "scipy",
]

setup(
    name="peft",
    version=VERSION,
    description="Parameter-Efficient Fine-Tuning (PEFT)",
    license_files=["LICENSE"],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="The HuggingFace team",
    author_email="sourab@huggingface.co",
    url="https://github.com/huggingface/peft",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"peft": ["py.typed"]},
    entry_points={},
    python_requires=">=3.8.0",
    install_requires=[
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.13.0",
        "transformers",
        "tqdm",
        "accelerate>=0.21.0",
        "safetensors",
        "huggingface_hub>=0.17.0",
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Release checklist
# 1. Change the version in __init__.py and setup.py to the release version, e.g. from "0.6.0.dev0" to "0.6.0"
# 2. Check if there are any deprecations that need to be addressed for this release by searching for "# TODO" in the code
# 3. Commit these changes with the message: "Release: VERSION", create a PR and merge it.
# 4. Add a tag in git to mark the release: "git tag -a VERSION -m 'Adds tag VERSION for pypi' "
#    Push the tag to git:
#      git push --tags origin main
#    It is necessary to work on the original repository, not on a fork.
# 5. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
#    Ensure that you are on the clean and up-to-date main branch (git status --untracked-files=no should not list any
#    files and show the main branch)
# 6. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
# 7. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi --extra-index-url https://pypi.org/simple peft
# 8. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 9. Add release notes to the tag on https://github.com/huggingface/peft/releases once everything is looking hunky-dory.
#      Check the notes here: https://docs.google.com/document/d/1k-sOIfykuKjWcOIALqjhFKz4amFEp-myeJUJEzNgjoU/edit?usp=sharing
# 10. Update the version in __init__.py, setup.py to the bumped minor version + ".dev0" (e.g. from "0.6.0" to "0.7.0.dev0")
