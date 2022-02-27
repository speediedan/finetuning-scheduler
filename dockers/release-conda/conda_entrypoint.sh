#!/bin/bash --login
set -e
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
. ~/.bashrc
exec "$@"
