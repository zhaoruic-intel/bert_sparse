#!/bin/bash
docker build --network=host . --rm --pull --no-cache -t bert --build-arg http_proxy=http://proxy-us.intel.com:911 --build-arg https_proxy=http://proxy-us.intel.com:912
