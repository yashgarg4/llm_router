# llmrouter — developer tasks.
# On Windows, run these under Git Bash (the shell Make invokes) or run the
# underlying command directly. PY points at the venv interpreter.

PY ?= .venv/Scripts/python.exe   # Unix venv: .venv/bin/python

.PHONY: install demo test lint clean demo-fallback proxy dashboard

install:            ## editable install (core deps)
	$(PY) -m pip install -e .

demo:               ## routing decision table (Phase 1)
	$(PY) examples/demo_routing.py

test:               ## run the test suite
	$(PY) -m pytest -q

# --- populated in later phases ---
demo-fallback:      ## forced-failure escalation demo (Phase 3)
	$(PY) examples/demo_fallback.py

proxy:              ## OpenAI-compatible proxy (Phase 5)
	$(PY) -m uvicorn server.proxy:app --reload

dashboard:          ## Streamlit per-route dashboard (Phase 4)
	$(PY) -m streamlit run dashboard/app.py

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info
