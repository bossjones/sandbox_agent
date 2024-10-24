set shell := ["zsh", "-cu"]

# just manual: https://github.com/casey/just/#readme

# Ignore the .env file that is only used by the web service
set dotenv-load := false

CURRENT_DIR := "$(pwd)"


# base64_cmd := if "{{os()}}" == "macos" { "base64 -w 0 -i cert.pem -o ca.pem" } else { "base64 -b 0 -i cert.pem -o ca.pem" }
base64_cmd := if "{{os()}}" == "macos" { "base64 -w 0 -i cert.pem -o ca.pem" } else { "base64 -w 0 -i cert.pem > ca.pem" }
grep_cmd := if "{{os()}}" =~ "macos" { "ggrep" } else { "grep" }

# List all available just commands
_default:
		@just --list

# Print the current operating system
info:
		print "OS: {{os()}}"

# Display system information
system-info:
	@echo "CPU architecture: {{ arch() }}"
	@echo "Operating system type: {{ os_family() }}"
	@echo "Operating system: {{ os() }}"

# verify python is running under pyenv
which-python:
		python -c "import sys;print(sys.executable)"

# when developing, you can use this to watch for changes and restart the server
autoreload-code:
	rye run watchmedo auto-restart --pattern "*.py" --recursive --signal SIGTERM rye run goobctl go

# Open the HTML coverage report in the default
local-open-coverage:
	./scripts/open-browser.py file://${PWD}/htmlcov/index.html

# Open the HTML coverage report in the default
open-coverage: local-open-coverage

# Run unit tests and open the coverage report
local-unittest:
	bash scripts/unittest-local
	./scripts/open-browser.py file://${PWD}/htmlcov/index.html

# Fetch multiple Python versions using rye
rye-get-pythons:
	rye fetch 3.8.19
	rye fetch 3.9.19
	rye fetch 3.10.14
	rye fetch 3.11.4
	rye fetch 3.12.3

# Add all dependencies using a custom script
rye-add-all:
	./contrib/rye-add-all.sh

# Run all pre-commit hooks on all files
pre-commit-run-all:
	pre-commit run --all-files

# Install pre-commit hooks
pre-commit-install:
	pre-commit install

# Display the dependency tree of the project
pipdep-tree:
	pipdeptree --python .venv/bin/python3

# install rye tools globally
rye-tool-install:
	rye install invoke
	rye install pipdeptree
	rye install click

# Lint GitHub Actions workflow files
lint-github-actions:
	actionlint

# check that taplo is installed to lint/format TOML
check-taplo-installed:
	@command -v taplo >/dev/null 2>&1 || { echo >&2 "taplo is required but it's not installed. run 'brew install taplo'"; exit 1; }

# Format Python files using pre-commit
fmt-python:
	git ls-files '*.py' '*.ipynb' | xargs rye run pre-commit run --files

# Format Markdown files using pre-commit
fmt-markdown-pre-commit:
	git ls-files '*.md' | xargs rye run pre-commit run --files

# format pyproject.toml using taplo
fmt-toml:
	pre-commit run taplo-format --all-files

# SOURCE: https://github.com/PovertyAction/ipa-data-tech-handbook/blob/ed81492f3917ee8c87f5d8a60a92599a324f2ded/Justfile

# Format all markdown and config files
fmt-markdown:
	git ls-files '*.md' | xargs rye run mdformat

# Format a single markdown file, "f"
fmt-md f:
	rye run mdformat {{ f }}

# format all code using pre-commit config
fmt: fmt-python fmt-toml fmt-markdown fmt-markdown fmt-markdown-pre-commit

# lint python files using ruff
lint-python:
	pre-commit run ruff --all-files

# lint TOML files using taplo
lint-toml: check-taplo-installed
	pre-commit run taplo-lint --all-files

# lint yaml files using yamlfix
lint-yaml:
	pre-commit run yamlfix --all-files

# lint pyproject.toml and detect log_cli = true
lint-check-log-cli:
	pre-commit run detect-pytest-live-log --all-files

# Check format of all markdown files
lint-check-markdown:
	rye run mdformat --check .

# Lint all files in the current directory (and any subdirectories).
lint: lint-python lint-toml lint-check-log-cli lint-check-markdown

# SOURCE: https://github.com/RobertCraigie/prisma-client-py/blob/da53c4280756f1a9bddc3407aa3b5f296aa8cc10/Makefile#L77
# Remove all generated files and caches
clean:
	rm -rf .cache
	rm -rf `find . -name __pycache__`
	rm -rf .tests_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	rm -f coverage.xml

# generate type stubs for the project
createstubs:
	./scripts/createstubs.sh

# sweep init
sweep-init:
	rye run sweep init

# TODO: We should try out trunk
# By default, we use the following config that runs Trunk, an opinionated super-linter that installs all the common formatters and linters for your codebase. You can set up and configure Trunk for yourself by following https://docs.trunk.io/get-started.
# sandbox:
#   install:
#     - trunk init
#   check:
#     - trunk fmt {file_path}
#     - trunk check {file_path}

# Download AI models from Dropbox
download-models:
	curl -L 'https://www.dropbox.com/s/im6ytahqgbpyjvw/ScreenNetV1.pth?dl=1' > src/sandbox_agent/data/ScreenNetV1.pth

# Perform a dry run of dependency upgrades
upgrade-dry-run:
	rye lock --update-all --all-features

# Upgrade all dependencies and sync the environment
sync-upgrade-all:
	rye sync --update-all --all-features

# Start a background HTTP server for test fixtures
http-server-background:
	#!/bin/bash
	# _PID=$(pgrep -f " -m http.server --bind localhost 19000 -d ./tests/fixtures")
	pkill -f " -m http.server --bind localhost 19000 -d ./tests/fixtures"
	python3 -m http.server --bind localhost 19000 -d ./tests/fixtures &
	echo $! > PATH.PID

# Start an HTTP server for test fixtures
http-server:
	#!/bin/bash
	# _PID=$(pgrep -f " -m http.server --bind localhost 19000 -d ./tests/fixtures")
	pkill -f " -m http.server --bind localhost 19000 -d ./tests/fixtures"
	python3 -m http.server --bind localhost 19000 -d ./tests/fixtures
	echo $! > PATH.PID

# Bump the version by major
major-version-bump:
	rye version
	rye version --bump major

# Bump the version by minor
minor-version-bump:
	rye version
	rye version --bump minor

# Bump the version by patch
patch-version-bump:
	rye version
	rye version --bump patch

# Bump the version by major
version-bump-major: major-version-bump

# Bump the version by minor
version-bump-minor: minor-version-bump

# Bump the version by patch
version-bump-patch: patch-version-bump

# Serve the documentation locally for preview
docs_preview:
    rye run mkdocs serve

# Build the documentation
docs_build:
    rye run mkdocs build

# Deploy the documentation to GitHub Pages
docs_deploy:
    rye run mkdocs gh-deploy --clean

# Generate a draft changelog
changelog:
    rye run towncrier build --version main --draft

# Checkout main branch and pull latest changes
gco:
    gco main
    git pull --rebase

# Show diff for LangChain migration
langchain-migrate-diff:
    langchain-cli migrate --include-ipynb --diff src

# Perform LangChain migration
langchain-migrate:
    langchain-cli migrate --include-ipynb src

get-ruff-config:
	rye run ruff check --show-settings --config pyproject.toml -v -o ruff_config.toml >> ruff.log 2>&1

ci:
	rye run lint
	rye run test

manhole-shell:
	./scripts/manhole-shell

find-cassettes-dirs:
	fd -td cassettes

delete-cassettes:
	fd -td cassettes -X rm -ri
# find tests -type d -name "*cassettes*" -print0 | xargs -0 -I {} rm -rfv {}


# regenerate-cassettes:
# 	fd -td cassettes -X rm -ri
# 	rye run unittests-vcr-record-final
# 	rye run unittests-debug

brew-deps:
	brew install libmagic poppler tesseract pandoc qpdf tesseract-lang
	brew install --cask libreoffice

db-create:
	rye run psql -d langchain -c 'CREATE EXTENSION vector'

# install aicommits and configure it
init-aicommits:
	npm install -g aicommits
	aicommits config set OPENAI_KEY=$OCO_OPENAI_API_KEY type=conventional model=gpt-4o-mini
	aicommits hook install

aider:
	aider -c .aider.conf.yml --aiderignore .aiderignore

aider-claude:
	aider -c .aider.conf.yml --aiderignore .aiderignore --model 'anthropic/claude-3-5-sonnet-20241022'

pur:
	cp -a .github/dependabot/requirements-dev.txt pur.before.txt
	rye run pur -d -r .github/dependabot/requirements-dev.txt > pur.after.txt
	diff pur.before.txt pur.after.txt | colordiff


# In order to properly create new cassette files, you must first delete the existing cassette files and directories. This will regenerate all cassette files and rerun tests.
delete-existing-cassettes:
	./scripts/delete-existing-cassettes.sh

# delete all cassette files and directories, regenerate all cassette files and rerun tests
local-regenerate-cassettes:
	@echo -e "\nDelete all cassette files and directories\n"
	./scripts/delete-existing-cassettes.sh
	@echo -e "\nRegenerate all cassette files using --record-mode=all\n"
	@echo -e "\nNOTE: This is expected to FAIL the first time when it is recording the cassette files!\n"
	rye run unittests-vcr-record-final || true
	@echo -e "\nrun regulate tests to verify that the cassettes are working\n"
	rye run ci-debug

# (alias) delete all cassette files and directories, regenerate all cassette files and rerun tests
local-regenerate-vcr: local-regenerate-cassettes

regenerate-cassettes: local-regenerate-cassettes

# This applies the ruff fixer to the target file and shows the fixes using a more aggressive formatter.
superfmt target:
	rye run ruff check --fix --show-fixes --select "E4,E7,E9,F,B,I,D,ERA" --fixable=ALL --unfixable="B003" --config=pyproject.toml {{ target }}

update-crucial-deps:
	rye lock --update aider-chat --update langchain-core --update langchain-community --update langgraph --update langsmith --update langsmith-community --update langsmith-core --update langsmith-server --update langchain-anthropic --update langchain-chroma --update langchain-google-genai --update langchain-groq --update langchain-openai --update langchain --update langchainhub --update 'langserve[all]' --all-features
	rye sync --all-features

dc-reset:
	docker compose down -v
	@docker network rm net || true
	@docker volume rm sbx_pgdata || true
	@docker volume rm sbx_pgadmindata || true
	@docker volume rm sbx_goob_redis_data || true
	sleep 30
	docker compose up -d
	./scripts/wait-until "docker compose exec -T -e PGPASSWORD=langchain pgdatabase psql -U langchain langchain -c 'select 1'" 300
	rye run db_upgrade

dc-reset-logs:
	docker compose down -v
	@docker network rm net || true
	@docker volume rm sbx_pgdata || true
	@docker volume rm sbx_pgadmindata || true
	@docker volume rm sbx_goob_redis_data || true
	sleep 30
	docker compose up -d
	./scripts/wait-until "docker compose exec -T -e PGPASSWORD=langchain pgdatabase psql -U langchain langchain -c 'select 1'" 300
	rye run db_upgrade
	docker compose logs -f | ccze -A

dc-postgres-tuning:
	docker compose exec -T -e PGPASSWORD=langchain pgdatabase psql -U langchain langchain -c max_connections=40

# -c 'shared_buffers=1GB' -c 'effective_cache_size=3GB' -c 'maintenance_work_mem=512MB' -c 'checkpoint_completion_target=0.9' -c 'wal_buffers=16MB' -c 'default_statistics_target=500' -c 'random_page_cost=1.1' -c 'effective_io_concurrency=200' -c 'work_mem=6553kB' -c 'huge_pages=off' -c 'min_wal_size=4GB' -c 'max_wal_size=16GB'

nltk-download:
	rye run python -m nltk.downloader popular
	rye run python -m nltk.downloader punkt_tab

wait-until-postgres-ready:
	./scripts/wait-until "docker compose exec -T -e PGPASSWORD=langchain pgdatabase psql -U langchain langchain -c 'select 1'" 300
