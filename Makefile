# llmrouter — developer tasks.
# On Windows, run these under Git Bash (the shell Make invokes) or run the
# underlying command directly. PY points at the venv interpreter.

PY ?= .venv/Scripts/python.exe   # Unix venv: .venv/bin/python

.PHONY: install demo test lint clean demo-fallback demo-semcache proxy proxy-metrics dashboard

install:            ## editable install (core deps)
	$(PY) -m pip install -e .

demo:               ## routing decision table (Phase 1-2)
	$(PY) examples/demo_routing.py

demo-fallback:      ## forced-failure escalation demo (Phase 3)
	$(PY) examples/demo_fallback.py

demo-semcache:      ## cache-in-front-of-router demo, real Gemini (Phase 5)
	$(PY) examples/demo_with_semcache.py

test:               ## run the test suite
	$(PY) -m pytest -q

proxy:              ## OpenAI-compatible proxy (Phase 5)
	$(PY) -m uvicorn server.proxy:app --reload

proxy-metrics:      ## metrics REST API (Phase 4): /metrics /by-route /alerts
	$(PY) -m uvicorn server.dashboard:app --reload

dashboard:          ## Streamlit per-route dashboard (Phase 4)
	$(PY) -m streamlit run dashboard/app.py

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info
