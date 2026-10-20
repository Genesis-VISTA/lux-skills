.PHONY: help lint policy test test-cov verify-new-skill

help:
	@echo "Available targets:"
	@echo "  make lint    Run Python syntax checks and the repo hygiene hooks"
	@echo "  make policy  Validate repo-level license and attribution policy files"
	@echo "  make test     Run the root smoke tests and tests with unittest"
	@echo "  make test-cov Run the same tests under trace for a line-count summary"
	@echo "  make verify-new-skill Validate a new or updated skill before opening a PR"

lint:
	python3 -m compileall -q tools tests
	@command -v pre-commit >/dev/null 2>&1 || { \
		echo "pre-commit is not installed; run 'python3 -m pip install pre-commit' before 'make lint'."; \
		exit 127; \
	}
	pre-commit run --all-files --show-diff-on-failure

policy:
	python3 tools/validate_repo_policy.py

test:
	python3 tools/run_tests.py

test-cov:
	python3 tools/run_tests.py --coverage

verify-new-skill:
	python3 tools/validate_skills.py --profile all --no-color --format text
	$(MAKE) policy
	$(MAKE) test
	$(MAKE) lint
