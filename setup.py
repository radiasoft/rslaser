# -*- coding: utf-8 -*-
"""rslaser setup script

:copyright: Copyright (c) 2020-2022 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
from pykern import pksetup

pksetup.setup(
    name="rslaser",
    author="RadiaSoft LLC",
    author_email="pip@radiasoft.net",
    description="A Python library for modeling chirped pulse amplifiers in high-power short-pulse lasers",
    install_requires=[
        "rsmath@git+https://github.com/radiasoft/rsmath.git",
        "matplotlib",
        "numpy",
        "pykern",
        "scipy",
    ],
    license="http://www.apache.org/licenses/LICENSE-2.0.html",
    url="https://github.com/radiasoft/rslaser",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python",
        "Topic :: Utilities",
    ],
    include_package_data=True,
)
