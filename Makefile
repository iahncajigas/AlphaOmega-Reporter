.PHONY: lint macos-build test windows-build

macos-build:
	./scripts/build_macos.sh

windows-build:
	pwsh -File ./scripts/build_windows.ps1

test:
	pytest -q

lint:
	ruff format --check .
	ruff check .
