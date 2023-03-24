#!/bin/bash
set -eou pipefail
pykern fmt diff rslaser_new tests setup.py
pykern test
