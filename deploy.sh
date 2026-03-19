#!/bin/bash
# Mirror TEP-JWST deploy script
npm run build:markdown --prefix site
cp -r site/dist/* /path/to/deployment/destination/ # Update this later
