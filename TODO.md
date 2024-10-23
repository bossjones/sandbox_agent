# TODO

### Start here

```
{
  "dependencies": ["./src/sandbox_agent"],
  "graphs": {
    "agent": "./src/sandbox_agent/ai/workflows/basic.py:graph"
  },
  "env": "./.env.langraph",
  "python_version": "3.12",
  "dockerfile_lines": [
    "RUN apt-get update",
    "RUN apt-get -y install --no-install-recommends python3-dev python3 ca-certificates python3-numpy python3-setuptools python3-wheel python3-pip g++ gcc ninja-build cmake",
    "RUN apt-get install -y --no-install-recommends aria2 aria2",
    "RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential gcc g++ cmake autoconf",
    "RUN apt-get install -y libmagic-dev poppler-utils libreoffice libomp-dev",
    "RUN apt-get install -y tesseract-ocr tesseract-ocr-por libyaml-dev poppler-utils",
    "RUN apt install ffmpeg -y",
    "RUN apt-get install autoconf automake build-essential libtool python3-dev libsqlite3-dev -y",
    "RUN apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git",
    "RUN pip install uv"
  ]
}

```

### Start here tomorrow (2024-10-15)

- https://github.com/langchain-ai/opengpts/blob/main/docker-compose.yml


### Start here tomorrow (2024-09-02)

- https://github.com/apify/actor-vector-database-integrations (need classes similar to this, and as configurable as this
- add some of the github actions from here
    https://github.com/genai-musings/template-repo-template/tree/main/.github/workflows
- https://github.com/benman1/generative_ai_with_langchain/tree/main/prompting
- https://github.com/kyrolabs/awesome-langchain
