# # note: this is not needed at the moment, but it's a good idea to have it.
# # TODO: Add script to install rye and use it to install the dependencies.
# # TODO: Add script to install asdf-vm/asdf and use it to install the dependencies.
# FROM debian:bullseye-slim

# ENV PYTHONDONTWRITEBYTECODE=1

# # ENV DEBIAN_FRONTEND=noninteractive
# # # LABEL mantainer="Read the Docs <support@readthedocs.com>"
# # # LABEL version="ubuntu-22.04-2022.03.15"

# # # ENV DEBIAN_FRONTEND noninteractive
# # ENV LANG C.UTF-8

# # RUN apt update && apt install -y curl git gnupg zsh tar software-properties-common vim


# # # Running this here so we can add tools quickly while relying on cache for layers above
# # RUN apt update && apt install -yq fzf perl gettext direnv vim awscli

# # # SOURCE: https://github.com/readthedocs/readthedocs-docker-images/blob/main/Dockerfile
# # # Install requirements
# # RUN apt-get -y install \
# #         build-essential \
# #         bzr \
# #         curl \
# #         doxygen \
# #         g++ \
# #         git-core \
# #         graphviz-dev \
# #         libbz2-dev \
# #         libcairo2-dev \
# #         libenchant-2-2 \
# #         libevent-dev \
# #         libffi-dev \
# #         libfreetype6 \
# #         libfreetype6-dev \
# #         libgraphviz-dev \
# #         libjpeg8-dev \
# #         libjpeg-dev \
# #         liblcms2-dev \
# #         libmysqlclient-dev \
# #         libpq-dev \
# #         libreadline-dev \
# #         libsqlite3-dev \
# #         libtiff5-dev \
# #         libwebp-dev \
# #         libxml2-dev \
# #         libxslt1-dev \
# #         libxslt-dev \
# #         mercurial \
# #         pandoc \
# #         pkg-config \
# #         postgresql-client \
# #         subversion \
# #         zlib1g-dev

# # # # LaTeX -- split to reduce image layer size
# # # RUN apt-get -y install \
# # #     texlive-fonts-extra
# # # RUN apt-get -y install \
# # #     texlive-lang-english
# # # RUN apt-get -y install \
# # #     texlive-full

# # # asdf Python extra requirements
# # # https://github.com/pyenv/pyenv/wiki#suggested-build-environment
# # RUN apt-get install -y \
# #     liblzma-dev \
# #     libncursesw5-dev \
# #     libssl-dev \
# #     libxmlsec1-dev \
# #     llvm \
# #     make \
# #     tk-dev \
# #     wget \
# #     xz-utils

# # # asdf nodejs extra requirements
# # # https://github.com/asdf-vm/asdf-nodejs#linux-debian
# # RUN apt-get install -y \
# #     dirmngr \
# #     gpg

# # # asdf Golang extra requirements
# # # https://github.com/kennyp/asdf-golang#linux-debian
# # RUN apt-get install -y \
# #     coreutils

# RUN apt-get update && apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
#     libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
#     mecab-ipadic-utf8 git ca-certificates vim zsh ffmpeg imagemagick coreutils

# ENV HOME /root
# ENV PYENV_ROOT $HOME/.pyenv
# ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
# RUN git clone --depth=1 https://github.com/pyenv/pyenv.git $PYENV_ROOT
# RUN git clone --depth=1 https://github.com/pyenv/pyenv-virtualenv.git $PYENV_ROOT/plugins/pyenv-virtualenv

# RUN pyenv install 3.9.20 && pyenv install 3.10.15 && pyenv install 3.11.10 && pyenv install 3.12.7 && pyenv install 3.13.0 && pyenv global 3.12.7

# RUN pip install uv

# WORKDIR /app
# COPY requirements.lock ./
# COPY requirements-dev.lock ./
# RUN uv pip install --no-cache --system -r requirements-dev.lock

# COPY src .




# # # ############################################################################################################################
# # # Install asdf
# # # ############################################################################################################################
# # RUN git clone https://github.com/asdf-vm/asdf.git ~/.asdf --depth 1 --branch v0.10.2
# # RUN echo ". /root/.asdf/asdf.sh" >> /root/.bashrc
# # RUN echo ". /root/.asdf/completions/asdf.bash" >> /root/.bashrc

# # # Activate asdf in current session
# # ENV PATH /root/.asdf/shims:/root/.asdf/bin:$PATH

# # # Install asdf plugins
# # RUN asdf plugin add python
# # RUN asdf plugin add nodejs https://github.com/asdf-vm/asdf-nodejs.git
# # RUN asdf plugin add rust https://github.com/code-lever/asdf-rust.git
# # RUN asdf plugin add golang https://github.com/kennyp/asdf-golang.git
# # # RUN asdf plugin-add hadolint https://github.com/looztra/asdf-hadolint
# # # RUN asdf plugin add fd
# # RUN asdf plugin-add tmux https://github.com/aphecetche/asdf-tmux.git
# # # RUN asdf plugin-add helm https://github.com/Antiarchitect/asdf-helm.git
# # # RUN asdf plugin-add jsonnet https://github.com/Banno/asdf-jsonnet.git
# # # RUN asdf plugin-add k9s https://github.com/looztra/asdf-k9s
# # # RUN asdf plugin-add kubectl https://github.com/Banno/asdf-kubectl.git
# # # RUN asdf plugin add kubectx
# # RUN asdf plugin-add neovim
# # # RUN asdf plugin-add packer https://github.com/Banno/asdf-hashicorp.git
# # # RUN asdf plugin-add terraform https://github.com/Banno/asdf-hashicorp.git
# # # RUN asdf plugin-add vault https://github.com/Banno/asdf-hashicorp.git
# # # RUN asdf plugin-add poetry https://github.com/crflynn/asdf-poetry.git
# # # RUN asdf plugin-add yq https://github.com/sudermanjr/asdf-yq.git
# # # RUN asdf plugin add ag https://github.com/koketani/asdf-ag.git
# # RUN asdf plugin-add aria2 https://github.com/asdf-community/asdf-aria2.git
# # # RUN asdf plugin-add argo https://github.com/sudermanjr/asdf-argo.git
# # # RUN asdf plugin-add dive https://github.com/looztra/asdf-dive
# # RUN asdf plugin-add github-cli https://github.com/bartlomiejdanek/asdf-github-cli.git
# # # RUN asdf plugin add kompose
# # RUN asdf plugin add mkcert
# # RUN asdf plugin-add shellcheck
# # RUN asdf plugin-add shfmt
# # # RUN asdf plugin-add velero https://github.com/looztra/asdf-velero

# # # Create directories for languages installations
# # RUN mkdir -p /root/.asdf/installs/python && \
# #     mkdir -p /root/.asdf/installs/nodejs && \
# #     mkdir -p /root/.asdf/installs/rust && \
# #     mkdir -p /root/.asdf/installs/golang && \
# #     mkdir -p /root/.asdf/installs/neovim && \
# #     mkdir -p /root/.asdf/installs/shfmt && \
# #     mkdir -p /root/.asdf/installs/shellcheck && \
# #     mkdir -p /root/.asdf/installs/mkcert && \
# #     mkdir -p /root/.asdf/installs/github-cli && \
# #     mkdir -p /root/.asdf/installs/yq && \
# #     mkdir -p /root/.asdf/installs/tmux

# # RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# # RUN curl --proto '=https' -fLsS https://rossmacarthur.github.io/install/crate.sh \
# #     | bash -s -- --repo rossmacarthur/sheldon --to ~/.local/bin

# # # && \
# # #     ~/.local/bin/sheldon init --shell zsh


# # ENV PATH /root/bin:/root/.bin:/root/.local/bin:$PATH

# # RUN asdf install python 3.9.13 && \
# #     asdf global python 3.9.13 && \
# #     asdf install golang 1.16.15 && \
# #     asdf global golang 1.16.15

# # COPY plugins.toml /root/.sheldon/plugins.toml
# # RUN sheldon lock
# # COPY zshrc.sheldon /root/.zshrc
# # # COPY zshrc /root/.zshrc
# # COPY asdf.sh /install-asdf.sh

# # RUN apt-get install unzip autotools-dev automake pkg-config libpcre3-dev zlib1g-dev liblzma-dev silversearcher-ag ripgrep locales -y

# # RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment && \
# #     echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
# #     echo "LANG=en_US.UTF-8" > /etc/locale.conf && \
# #     locale-gen en_US.UTF-8

# # RUN asdf install neovim 0.7.2 && \
# #     asdf global neovim 0.7.2 && \
# #     asdf install aria2 1.36.0 && \
# #     asdf global aria2 1.36.0 && \
# #     asdf install github-cli 2.0.0 && \
# #     asdf global github-cli 2.0.0 && \
# #     asdf install shellcheck 0.8.0 && \
# #     asdf global shellcheck 0.8.0 && \
# #     asdf install nodejs 16.16.0 && \
# #     asdf global nodejs 16.16.0 && \
# #     asdf install shfmt 3.3.1 && \
# #     asdf global shfmt 3.3.1
