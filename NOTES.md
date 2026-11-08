# Submission notes

- Python 3.9+; standard library only.
- No API keys or external services are used.
- `answer(question, as_of=...)` selects the newest ingested snapshot whose
  `taken_on` date is not later than `as_of`. Without a date, it uses the newest
  snapshot.
- Answers cite the source URL and snapshot. `conflicts()` reports changing
  rate-table rows, their observed values, the newest value, and the reason for
  preferring it.
