#!/bin/bash
set -eou pipefail
pykern fmt diff rslaser tests setup.py
pykern test
