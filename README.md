# BankGPT knowledge system

A standard-library-only, temporal retrieval implementation for the interface.ai
knowledge-system build assignment. The submission contract is implemented in
[`solution.py`](./solution.py): `build()` returns an object with `ingest`,
`answer`, and `conflicts` methods.

The knowledge system indexes each supplied snapshot independently, selects the
latest snapshot on or before `as_of`, ranks source passages using query
relevance and page authority, and cites the page used. Broad questions that
request a list can return multiple relevant passages; unsupported rate
qualifiers are answered with an explicit refusal rather than a nearby rate.

No API keys or third-party packages are required. The participant kit's
harness, public evaluation questions, and starter code are kept in
[`hackathon-knowledge-kit-participant/`](./hackathon-knowledge-kit-participant/).
The supplied assignment PDF and website snapshot data remain local and are
excluded from this public repository.

Run the public evaluation from the repository root with:

```bash
python hackathon-knowledge-kit-participant/harness/run_eval.py \
  --submission . \
  --snapshots hackathon-knowledge-kit-participant/snapshots \
  --eval hackathon-knowledge-kit-participant/eval/public_eval.jsonl
```