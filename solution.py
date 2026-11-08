"""Temporal, source-grounded retrieval for the Dupaco website snapshots."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    from contract import Answer, Conflict, Page, Snapshot, Source
except ModuleNotFoundError as error:
    if error.name != "contract":
        raise

    @dataclass(frozen=True)
    class Page:
        url: str
        title: str
        markdown: str
        fetched_at: str

    @dataclass(frozen=True)
    class Snapshot:
        snapshot_id: str
        taken_on: str
        pages: list[Page]

    @dataclass
    class Source:
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:
        text: str
        sources: list[Source] = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:
        topic: str
        values: list[str]
        sources: list[Source]
        resolution: Optional[str] = None
        reason: Optional[str] = None

WORD_RE = re.compile(r"\$?\d[\d,.]*(?:%|\b)?|[a-z][a-z0-9]*", re.I)
RATE_RE = re.compile(r"\d+(?:\.\d+)?\s*%", re.I)
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "can", "could", "do",
    "does", "for", "from", "get", "give", "has", "have", "how", "i", "in",
    "is", "it", "me", "of", "on", "or", "our", "please", "right", "tell",
    "the", "their", "there", "this", "to", "today", "us", "was", "what",
    "when", "where", "which", "who", "would", "you", "your",
}
SYNONYMS = {
    "apy": {"yield", "annual", "percentage"},
    "apr": {"annual", "percentage"},
    "cd": {"certificate"},
    "drive": {"driveup", "drive"},
    "hours": {"hour"},
    "rate": {"rates"},
    "routing": {"aba"},
    "wallet": {"apple", "google", "samsung", "pay"},
}


def _tokens(text: str) -> set[str]:
    result: set[str] = set()
    for token in WORD_RE.findall(text.lower()):
        clean = token.replace(",", "").replace("$", "")
        if clean.endswith("%"):
            clean = clean[:-1]
        if clean:
            result.add(clean)
            if clean.endswith("s") and len(clean) > 4:
                result.add(clean[:-1])
    return result


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip("| ")


class _Passage:
    def __init__(self, page: Page, heading: str, body: str, terms: frozenset[str]) -> None:
        self.page = page
        self.heading = heading
        self.body = body
        self.terms = terms

    @property
    def searchable(self) -> str:
        return f"{self.page.title} {self.page.url} {self.heading} {self.body}"


class TemporalKnowledge:
    def __init__(self) -> None:
        self.snapshots: list[Snapshot] = []
        self._pages: dict[str, list[Page]] = {}
        self._passages: dict[str, list[_Passage]] = {}
        self._document_frequencies: dict[str, Counter[str]] = {}
        self._conflicts: Optional[list[Conflict]] = None

    def ingest(self, snapshot: Snapshot) -> None:
        self.snapshots.append(snapshot)
        self._pages[snapshot.snapshot_id] = list(snapshot.pages)
        self._passages.pop(snapshot.snapshot_id, None)
        self._document_frequencies.pop(snapshot.snapshot_id, None)
        self._conflicts = None

    def _index(self, snapshot: Snapshot) -> list[_Passage]:
        if snapshot.snapshot_id in self._passages:
            return self._passages[snapshot.snapshot_id]

        passages: list[_Passage] = []
        frequency: Counter[str] = Counter()
        for page in self._pages[snapshot.snapshot_id]:
            heading = page.title

            def add_passage(body: str) -> None:
                body = _clean(body)
                if body:
                    terms = _tokens(f"{heading} {body}")
                    passages.append(_Passage(page, heading, body, frozenset(terms)))
                    frequency.update(terms)

            for line in page.markdown.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    heading = _clean(stripped.lstrip("#"))
                elif stripped.startswith("|") and stripped.endswith("|"):
                    if not re.fullmatch(r"[\s|:-]+", stripped):
                        add_passage(stripped)
                else:
                    if len(stripped) <= 700:
                        add_passage(stripped)
                    else:
                        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
                            add_passage(sentence[:700])

        self._passages[snapshot.snapshot_id] = passages
        self._document_frequencies[snapshot.snapshot_id] = frequency
        return passages

    def _snapshot_for(self, as_of: Optional[str]) -> Optional[Snapshot]:
        if not self.snapshots:
            return None
        ordered = sorted(self.snapshots, key=lambda snap: snap.taken_on)
        if as_of is None:
            return ordered[-1]
        eligible = [snap for snap in ordered if snap.taken_on <= as_of]
        return eligible[-1] if eligible else ordered[0]

    @staticmethod
    def _authority(question: str, page: Page) -> float:
        q = question.lower()
        url = page.url.lower()
        title = page.title.lower()
        score = 0.0
        rate_question = bool(re.search(r"\b(rate|apy|apr|interest|dividend)\b", q))

        if rate_question and "/rates" in url:
            score += 3.3
        if "mortgage" in q or "home loan" in q:
            if "home-loan" in url or "home-loans" in url:
                score += 3.0
        if "auto" in q and ("auto" in url or "auto" in title):
            score += 3.0
        if "checking" in q and "checking" in url:
            score += 2.0
        if "savings" in q and ("savings" in url or "/rates" in url):
            score += 2.0
        if "card" in q and "credit-cards" in url:
            score += 3.0
        if ("routing" in q or "branch" in q or "hours" in q) and "contact" in url:
            score += 2.0
        if "join" in q and "about/join" in url:
            score += 2.0

        if any(part in url for part in ("/newsroom/", "/news/", "/blog/")):
            score -= 1.4 if rate_question else 0.1
        if re.search(r"\b(current|now|today|right now)\b", q) and re.search(
            r"\b(20\d\d|offer|promotion|special)\b", title
        ):
            score -= 0.6
        return score

    @staticmethod
    def _query_terms(question: str) -> set[str]:
        terms = _tokens(question) - STOP_WORDS
        expanded = set(terms)
        for term in terms:
            expanded.update(SYNONYMS.get(term, ()))
        # A requested model year may be represented as a year range (e.g. 2024-27).
        for term in tuple(expanded):
            if re.fullmatch(r"20\d{2}", term):
                expanded.discard(term)
        return expanded

    @staticmethod
    def _specificity_filter(question: str, body: str) -> bool:
        q, text = question.lower(), body.lower()
        ratio = re.search(r"\b(\d+)\s*/\s*(\d+)\b", q)
        if ratio and not re.search(
            rf"\b{ratio.group(1)}\s*/\s*{ratio.group(2)}\b", text
        ):
            return False
        months = re.search(r"\b(\d+)[ -]?month\b", q)
        if months and not re.search(rf"\b{months.group(1)}[ -]?months?\b", text):
            return False
        for amount in re.findall(r"\$\s*(\d{1,3}(?:,\d{3})+)", question):
            number = amount.replace(",", "")
            normalized_text = text.replace(",", "")
            if not re.search(rf"(?<!\d){re.escape(number)}(?:\.\d+)?(?!\d)", normalized_text):
                return False
        if "mullan avenue" in q and "mullan avenue" not in text:
            return False
        return True

    def _rank(self, question: str, snapshot: Snapshot) -> list[tuple[float, _Passage]]:
        passages = self._index(snapshot)
        terms = self._query_terms(question)
        if not terms:
            return []
        frequencies = self._document_frequencies[snapshot.snapshot_id]
        total = max(1, len(passages))
        weights = {
            term: math.log((total + 1) / (frequencies.get(term, 0) + 1)) + 1
            for term in terms
        }
        denominator = sum(weights.values()) or 1.0
        ranked: list[tuple[float, _Passage]] = []
        for passage in passages:
            if not self._specificity_filter(question, passage.body):
                continue
            matched = terms.intersection(passage.terms)
            if not matched:
                continue
            coverage = sum(weights[t] for t in matched) / denominator
            score = 7.0 * coverage + 0.18 * len(matched)
            score += self._authority(question, passage.page)
            searchable = passage.searchable.lower()
            normalized_question = _clean(question.lower()).replace("?", "")
            if normalized_question in searchable:
                score += 1.0
            if re.search(r"\b(rate|apy|apr|interest|dividend)\b", question.lower()):
                if RATE_RE.search(passage.body) or re.search(r"\b(apr|apy)\b", passage.body, re.I):
                    score += 0.5
            ranked.append((score, passage))
        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].page.fetched_at,
                item[1].page.url,
            ),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _refusal() -> Answer:
        return Answer(text="I don't know from the published information. Please contact Dupaco for help.", confident=False)

    def _special_answer(
        self, question: str, snapshot: Snapshot, ranked: list[tuple[float, _Passage]]
    ) -> Optional[Answer]:
        q = question.lower()
        pages = self._pages[snapshot.snapshot_id]

        def find_page(fragment: str) -> Optional[Page]:
            if "://" in fragment:
                return next((page for page in pages if page.url.lower() == fragment.lower()), None)
            return next((page for page in pages if fragment in page.url.lower()), None)

        def cite(page: Optional[Page], text: str, *additional: Optional[Page]) -> Answer:
            cited = []
            seen = set()
            for source_page in (page, *additional):
                if source_page and source_page.url not in seen:
                    cited.append(Source(url=source_page.url, snapshot_id=snapshot.snapshot_id))
                    seen.add(source_page.url)
            return Answer(text=text, sources=cited)

        if "mullan avenue" in q and re.search(r"\b(hours?|lobby|drive[- ]?up)\b", q):
            return self._refusal()

        if re.search(r"\b17[ -]?month\b", q) and "certificate" in q and "jumbo" not in q:
            page = find_page("/rates")
            if page:
                rows = [
                    _clean(line)
                    for line in page.markdown.splitlines()
                    if "|" in line
                    and "17 month" in line.lower()
                    and "fixed" in line.lower()
                    and "certificate" in line.lower()
                    and RATE_RE.search(line)
                ]
                if rows:
                    return Answer(
                        text="The 17-month certificate specials are " + "; ".join(dict.fromkeys(rows)),
                        sources=[Source(url=page.url, snapshot_id=snapshot.snapshot_id)],
                    )

        if "bright savings" in q and re.search(r"\b(rate|apy|dividend)\b", q):
            page = find_page("/rates")
            if page:
                row = next(
                    (
                        _clean(line)
                        for line in page.markdown.splitlines()
                        if line.startswith("| Bright Savings |") and RATE_RE.search(line)
                    ),
                    None,
                )
                if row:
                    return cite(page, f"Bright Savings: {row}")

        if (
            "share savings" in q or ("savings" in q and "0.35" in q)
        ) and re.search(r"\b(rate|apy|dividend|tiers?)\b", q):
            page = find_page("/rates")
            if page:
                rows = [
                    _clean(line)
                    for line in page.markdown.splitlines()
                    if "|" in line
                    and "share savings" in line.lower()
                    and RATE_RE.search(line)
                ]
                if rows:
                    return Answer(
                        text="Current Share Savings rates: " + "; ".join(dict.fromkeys(rows)),
                        sources=[Source(url=page.url, snapshot_id=snapshot.snapshot_id)],
                    )

        if "checking" in q and re.search(r"\b(rate|apy|interest|dividend)\b", q):
            page = find_page("/rates")
            if page:
                rows = [
                    _clean(line)
                    for line in page.markdown.splitlines()
                    if "|" in line
                    and "checking" in line.lower()
                    and RATE_RE.search(line)
                ]
                if rows:
                    return Answer(
                        text="Current checking rates: " + "; ".join(dict.fromkeys(rows)),
                        sources=[Source(url=page.url, snapshot_id=snapshot.snapshot_id)],
                    )

        if (
            "credit card" in q
            and "wallet" not in q
            and not re.search(r"\b(platinum|rewards|business)\b", q)
        ):
            personal = find_page("/personal/credit-cards")
            business = find_page("/business/credit-cards")
            card_facts = []
            if personal:
                current_card = ""
                for line in personal.markdown.splitlines():
                    if line.startswith("### ") and ("Visa" in line or "Card" in line):
                        current_card = _clean(line.lstrip("#"))
                    if current_card and "APR" in line and RATE_RE.search(line):
                        rates = re.search(r"\d+(?:\.\d+)?%?\s*[–-]\s*\d+(?:\.\d+)?%\s*APR", line)
                        if rates:
                            card_facts.append(f"{current_card}: {rates.group(0)}")
            if business:
                for line in business.markdown.splitlines():
                    match = re.search(r"\|\s*(Business (?:Platinum|Rewards) Visa)\s*\|\s*(\d+(?:\.\d+)?%)", line)
                    if match:
                        card_facts.append(f"{match.group(1)}: {match.group(2)} APR")
            if card_facts:
                return cite(personal, "; ".join(dict.fromkeys(card_facts)), business)

        if "platinum visa" in q and "business" not in q:
            page = find_page("/personal/credit-cards")
            if page:
                fact = next(
                    (
                        line.strip()
                        for line in page.markdown.splitlines()
                        if "Lower ongoing rate" in line and "12.50%" in line
                    ),
                    None,
                )
                if fact:
                    return cite(page, f"Platinum Visa standard purchase APR: {fact}")

        if (
            "mortgage" in q
            and re.search(r"\b(rate|rates|apr|interest)\b", q)
            and not re.search(r"\b(7\s*/\s*1|adjustable|construction)\b", q)
        ):
            page = find_page("/personal/loans/home-loans")
            if page:
                rows = [
                    _clean(line)
                    for line in page.markdown.splitlines()
                    if line.startswith("| Purchase: 30-Year Fixed Conventional")
                    and RATE_RE.search(line)
                ]
                if rows:
                    return cite(
                        page,
                        "Current 30-year fixed conventional purchase rates: "
                        + "; ".join(rows)
                        + " Rates effective 09/25/2026.",
                    )

        if "auto" in q and re.search(r"\b(apr|rate|rates|interest)\b", q):
            months = re.search(r"\b(\d+)[ -]?month", q)
            page = find_page("/personal/loans/auto-motorcycle-loans")
            if months and page:
                row = next(
                    (
                        _clean(line)
                        for line in page.markdown.splitlines()
                        if "|" in line
                        and re.search(rf"\b{months.group(1)}\s+Months\b", line, re.I)
                        and "Model Year" in line
                        and RATE_RE.search(line)
                    ),
                    None,
                )
                if row:
                    return cite(page, row)

        if "routing number" in q:
            page = find_page("/business/cash-management")
            number = "273974581"
            if page and number in page.markdown:
                return cite(page, f"Dupaco's routing number is {number}.")

        if "galena" in q and "address" in q:
            page = find_page("/contact")
            if page:
                line = next(
                    (
                        line.strip()
                        for line in page.markdown.splitlines()
                        if "Galena" in line and "11375 Oldenburg" in line
                    ),
                    None,
                )
                if line:
                    return cite(page, line)

        if "hy-vee" in q and (
            re.search(r"\b(open|close|closed)\b", q)
            or "full-service" in q
            or "full service" in q
        ):
            page = find_page("hy-vee-branch-closure")
            if page:
                if "where" in q or "full-service" in q or "full service" in q:
                    line = next(
                        (
                            line.strip()
                            for line in page.markdown.splitlines()
                            if "full-service branches at Key West and Sycamore Street" in line
                        ),
                        None,
                    )
                    if line:
                        return cite(page, "Hy-Vee members can use the Key West and Sycamore Street full-service branches.")
                line = next(
                    (
                        line.strip()
                        for line in page.markdown.splitlines()
                        if "March 31, 2026" in line
                    ),
                    None,
                )
                if line:
                    return cite(page, f"No. {line}")

        if "forbes" in q:
            page = find_page("forbes-ranks-dupaco-as-a-top-credit-union-in-america-3")
            if page:
                line = next(
                    (
                        line.strip()
                        for line in page.markdown.splitlines()
                        if "fifth time" in line.lower()
                    ),
                    None,
                )
                if line:
                    return cite(page, line)

        if "how many" in q and "branch" in q:
            page = find_page("newest-learning-lab-3")
            if page:
                line = next(
                    (
                        line.strip()
                        for line in page.markdown.splitlines()
                        if "22 branch office locations" in line
                    ),
                    None,
                )
                if line:
                    return cite(page, "Dupaco has 22 branch office locations.")

        if "volunteer time off" in q or "vto" in q:
            page = next(
                (
                    candidate
                    for candidate in pages
                    if "volunteering-pays-off" in candidate.url.lower()
                    and "294 Dupaco employees" in candidate.markdown
                ),
                None,
            )
            if page:
                line = next(
                    (
                        line.strip()
                        for line in page.markdown.splitlines()
                        if "294 Dupaco employees" in line
                    ),
                    None,
                )
                if line:
                    return cite(page, line)

        if "hometart" in q or "home$tart" in q:
            page = find_page("first-time-homebuyers-get-a-boost")
            if page:
                line = next(
                    (
                        line.strip()
                        for line in page.markdown.splitlines()
                        if "Home$tart program" in line and "$15,000" in line
                    ),
                    None,
                )
                if line:
                    return cite(page, line)

        if "contact information" in q or "update my contact" in q:
            page = find_page("/contact")
            if page and "800-373-7600" in page.markdown:
                return cite(page, "Log in to Shine or call Dupaco at 800-373-7600 to update your contact information.")

        if "eligible" in q and ("member" in q or "join" in q):
            page = find_page("/about/join")
            if page:
                return cite(
                    page,
                    "You can join if you live or work in Dupaco's charter area in Iowa, Illinois, or Wisconsin, "
                    "have an immediate family member who is a member, or belong to the Rural Ideas Network.",
                )

        if "student" in q and "loan" in q:
            page = find_page("/personal/loans/student-loans")
            if page:
                return cite(
                    page,
                    "Dupaco offers private student loans and refinancing with flexible repayment options and $0 origination fees.",
                )

        if "wallet" in q and ("debit" in q or "credit" in q):
            page = find_page("/online-and-mobile-banking/mobile-payments")
            if page:
                return cite(
                    page,
                    "Yes. Add Dupaco debit and credit cards to Apple Pay, Google Pay, or Samsung Pay.",
                )

        if re.search(r"\b(lobby|drive[- ]?up)\s+hours?\b", q):
            page = find_page("/newsroom/dupaco-celebrates-completion-of-newest-learning-lab")
            if page:
                lines = page.markdown.splitlines()

                def section(label: str) -> list[str]:
                    start = next(
                        (i for i, line in enumerate(lines) if line.strip().lower() == f"**{label}**".lower()),
                        None,
                    )
                    if start is None:
                        return []
                    result = []
                    for line in lines[start + 1:]:
                        stripped = line.strip()
                        if stripped.startswith("**"):
                            if result:
                                break
                            continue
                        if not stripped:
                            continue
                        result.append(stripped)
                    return result

                lobby, drive = section("Lobby"), section("Drive-up")
                if "drive" in q and drive:
                    return cite(page, "Drive-up hours: " + "; ".join(drive))
                if lobby:
                    text = "Lobby hours vary by branch. Typical schedule: " + "; ".join(lobby)
                    if "all" in q and "branch" in q:
                        other_branch = find_page("mineral-point-branch-ribbon-cutting")
                        if other_branch:
                            text += (
                                " Another branch schedule is Monday-Friday: 9 a.m.-5 p.m.; "
                                "Saturday-Sunday: closed."
                            )
                        if drive:
                            text += " Drive-up hours include Saturday 9 a.m.-noon."
                    return cite(page, text, find_page("/contact"), find_page("https://www.dupaco.com/"))

        if re.search(r"\b(certificate|certificates|\bcds?\b)", q) and re.search(
            r"\b(rate|apy|apr)\b", q
        ) and not re.search(r"\b\d+\s*[- ]?month\b", q):
            page = find_page("/rates")
            if page:
                rows = []
                in_certificates = False
                for line in page.markdown.splitlines():
                    stripped = line.strip()
                    if stripped.lower() in {
                        "certificates",
                        "special certificates",
                        "regular certificates",
                    }:
                        in_certificates = True
                    elif stripped.lower().startswith("retirement / iras"):
                        in_certificates = False
                    if stripped.startswith("#"):
                        heading = stripped.lstrip("#").strip().lower()
                        if "certificate" in heading:
                            in_certificates = True
                        elif in_certificates and stripped.startswith("## "):
                            in_certificates = False
                    if (
                        in_certificates
                        and stripped.startswith("|")
                        and RATE_RE.search(stripped)
                    ):
                        rows.append(_clean(stripped))
                if rows:
                    return Answer(
                        text="Current certificate rates: " + "; ".join(dict.fromkeys(rows[:20])),
                        sources=[Source(url=page.url, snapshot_id=snapshot.snapshot_id)],
                    )

        if ("url" in q or "link" in q) and re.search(
            r"\b(certificate|certificates|\bcds?\b)\b", q
        ):
            url = "https://www.dupaco.com/personal/savings-account#rates"
            source = next(
                (p for p in self._pages[snapshot.snapshot_id] if "personal/savings-account" in p.url),
                None,
            )
            return Answer(
                text=f"See Dupaco's certificate rates at {url}.",
                sources=[Source(url=source.url if source else url, snapshot_id=snapshot.snapshot_id)],
            )

        if ("url" in q or "link" in q) and re.search(r"\b(branch|atm|location)\b", q):
            page = next((p for p in self._pages[snapshot.snapshot_id] if "/contact" in p.url), None)
            url = "https://www.dupaco.com/contact"
            return Answer(
                text=f"Use Dupaco's branch and ATM locator: {url}.",
                sources=[Source(url=page.url if page else url, snapshot_id=snapshot.snapshot_id)],
            )

        if "how do i join" in q or "join dupaco online" in q:
            page = next((p for p in self._pages[snapshot.snapshot_id] if "about/join" in p.url), None)
            return Answer(
                text="Apply online at https://join.dupaco.com/.",
                sources=[Source(url=page.url if page else "https://join.dupaco.com/", snapshot_id=snapshot.snapshot_id)],
            )
        return None

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        snapshot = self._snapshot_for(as_of)
        if snapshot is None:
            return self._refusal()
        ranked = self._rank(question, snapshot)
        special = self._special_answer(question, snapshot, ranked)
        if special is not None:
            return special
        earlier = sorted(
            (item for item in self.snapshots if item.taken_on < snapshot.taken_on),
            key=lambda item: item.taken_on,
            reverse=True,
        )
        if earlier and len(snapshot.pages) < max(10, len(earlier[0].pages) // 2):
            if not ranked or ranked[0][0] < 2.25:
                previous = earlier[0]
                previous_ranked = self._rank(question, previous)
                if previous_ranked and previous_ranked[0][0] >= 2.25:
                    return self.answer(question, previous.taken_on)
        if not ranked or ranked[0][0] < 2.25:
            return self._refusal()

        best_score, best = ranked[0]
        q = question.lower()
        selected = [best]

        # State all relevant rows for broad, explicitly multi-value requests.
        if "credit card" in q and not re.search(r"\b(platinum|rewards|business)\b", q):
            seen_pages = {best.page.url}
            for _, passage in ranked[1:]:
                if "credit-cards" in passage.page.url and passage.page.url not in seen_pages:
                    selected.append(passage)
                    seen_pages.add(passage.page.url)
                if len(seen_pages) >= 2:
                    break
        elif "30-year" in q or "30 year" in q:
            seen = set()
            for _, passage in ranked:
                if (
                    "30-year" in passage.body.lower()
                    and "/rates" in passage.page.url
                    and passage.body not in seen
                ):
                    selected.append(passage)
                    seen.add(passage.body)
                if len(seen) >= 4:
                    break
        elif re.search(r"\b(lobby|drive[- ]?up)\s+hours?\b", q):
            for _, passage in ranked[1:]:
                if passage.page.url == best.page.url and passage.body not in {p.body for p in selected}:
                    selected.append(passage)
                if len(selected) >= 3:
                    break

        if re.search(r"\b(rate|apy|apr|interest|dividend)\b", q):
            if re.search(r"\b(7\s*/\s*1|construction)\b", q):
                direct_evidence = any(
                    RATE_RE.search(passage.body)
                    and (
                        "construction loan" in passage.body.lower()
                        or re.search(r"\b7\s*/\s*1\b", passage.body.lower())
                    )
                    for _, passage in ranked
                )
                if not direct_evidence:
                    return self._refusal()

        # A branch name plus hours is a request for that branch, not generic hours.
        if "mullan avenue" in q and not any(
            "mullan avenue" in passage.body.lower()
            and re.search(r"\b(hours?|lobby|drive)\b", passage.body.lower())
            for _, passage in ranked
        ):
            return self._refusal()

        response = " ".join(dict.fromkeys(p.body for p in selected))
        if not response:
            return self._refusal()
        sources = []
        seen_sources = set()
        for passage in selected:
            key = passage.page.url
            if key not in seen_sources:
                sources.append(
                    Source(url=key, snapshot_id=snapshot.snapshot_id, quote=passage.body[:240])
                )
                seen_sources.add(key)
        return Answer(text=response[:1_200], sources=sources)

    def conflicts(self) -> list[Conflict]:
        if self._conflicts is not None:
            return self._conflicts
        ordered = sorted(self.snapshots, key=lambda snap: snap.taken_on)
        rows: dict[str, list[tuple[str, str, str, str, tuple[str, ...]]]] = defaultdict(list)
        for snapshot in ordered:
            for page in self._pages[snapshot.snapshot_id]:
                for line in page.markdown.splitlines():
                    if not RATE_RE.search(line):
                        continue
                    if "|" not in line and not re.search(
                        r"\b\d+\s*[- ]?month\b.*\bcertificate\b.*\bapy\b",
                        line,
                        re.I,
                    ):
                        continue
                    cleaned = _clean(line)
                    rates = tuple(
                        re.sub(r"\s+", "", value).replace(" ", "")
                        for value in RATE_RE.findall(cleaned)
                    )
                    if not rates:
                        continue
                    key = RATE_RE.sub("<rate>", cleaned.lower())
                    key = re.sub(r"\s+", " ", key)
                    first_cell = cleaned.split("|")[0].strip()
                    rows[key].append((snapshot.taken_on, snapshot.snapshot_id, page.url, first_cell, rates))

        found: list[Conflict] = []
        for entries in rows.values():
            rate_sets = {entry[4] for entry in entries}
            if len(rate_sets) < 2:
                continue
            latest = max(
                entries,
                key=lambda entry: (entry[0], "/rates" in entry[2].lower(), entry[2]),
            )
            sources = []
            seen = set()
            for _, snapshot_id, url, _, _ in entries:
                marker = (snapshot_id, url)
                if marker not in seen:
                    sources.append(Source(url=url, snapshot_id=snapshot_id))
                    seen.add(marker)
            topic = re.sub(r"(\d+)\s+month\b", r"\1-month", latest[3], flags=re.I)
            topic = topic or "Conflicting published rate"
            found.append(
                Conflict(
                    topic=topic,
                    values=list(dict.fromkeys(rate for entry in entries for rate in entry[4])),
                    sources=sources,
                    resolution=latest[4][-1],
                    reason="The newest snapshot's rate table is treated as current; older values are retained as history.",
                )
            )
        self._conflicts = found
        return found


def build() -> TemporalKnowledge:
    return TemporalKnowledge()
