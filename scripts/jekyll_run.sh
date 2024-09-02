#!/bin/bash

# borrowed this from aider.

# Run the Docker container
docker run \
       --rm \
       -v "$PWD/sandbox_agent/website:/site" \
       -p 4000:4000 \
       -e HISTFILE=/site/.bash_history \
       -it \
       my-jekyll-site

#       --entrypoint /bin/bash \
