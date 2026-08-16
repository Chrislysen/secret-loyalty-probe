"""Reachability gate: every URL the paper prints must resolve for a stranger with no credentials.

    python writeup/check_links.py

This exists because of a measured, expensive defect. The submitted version of the paper carried
exactly one link -- line 8, directly under the title, the most prominent position in the document --
and it returned 404, because the repository was private at judging time. Nothing in the build caught
it: the link was correct, the repository existed, and every check we ran was authenticated as the
owner, for whom it resolves. A reviewer clicked the one link we gave them and got nothing.

So the gate's whole design is the word UNAUTHENTICATED. It fetches with the standard library only
(the project's core dependency is numpy; adding `requests` to run a link check would be a worse
defect than the one it fixes), and it deliberately carries no identity:

  * no Authorization header is ever set, and GITHUB_TOKEN / GH_TOKEN are never read;
  * no `.netrc` is consulted -- urllib does not read one unless an auth handler is installed, and
    this module installs none, then asserts that none is installed;
  * no git config, no git credential helper, no `gh auth` -- none of them are on urllib's path at
    all, which is exactly the point: what this process can see is what a stranger can see.

Two failure modes are NOT the same thing and the gate keeps them apart, because conflating them is
how a link check becomes noise that everyone learns to ignore:

  404 / 403 / 5xx  -- we reached a server and it refused us. That is the defect. Exit 1.
  DNS / timeout / refused, on EVERY url and on the canaries too -- there is no network here. That is
                      an offline build, not a broken paper. Print a WARNING, exit 0.

Exit codes:
  0  every URL that points at our own repository returned 200 -- or there is no network at all
     (WARNING), or the check was explicitly skipped via SKIP_LINK_CHECK=1 (loud WARNING).
  1  at least one of our own URLs did not return 200 while the network was demonstrably up, or a
     URL was malformed / carried embedded credentials.

External links (arXiv, model cards, third-party repos) are reported with their status but never fail
the build: we do not control them, and a paper should not become unbuildable because someone else's
server is having a bad afternoon.

Environment:
  SKIP_LINK_CHECK=1     skip entirely (prints a loud warning; for offline work)
  LINK_CHECK_TIMEOUT=n  per-request timeout in seconds (default 10)
  OWN_REPO_URL_RE=...   override the "is this our own repo" pattern (default: this repo on GitHub)
"""
from __future__ import annotations

import os
import re
import socket
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent

# The documents whose links a reviewer can actually click. REPORT.md is optional -- it is the
# technical appendix and is not always present in a slimmed checkout.
DOCS = ["PAPER.md", "REPORT.md"]

# What counts as OURS, and therefore as a build-breaking 404 rather than someone else's outage.
# Hardcoded rather than read from `git remote`, on purpose: the question this gate answers is
# "does the URL PRINTED IN THE PAPER work", and deriving the answer from the local git config is the
# same mistake as checking the link while logged in -- it substitutes our environment for a
# stranger's. Case-insensitive; matches the bare repo URL, any sub-path, the .git form, and Pages.
OWN_REPO_URL_RE = os.environ.get("OWN_REPO_URL_RE") or (
    r"^https?://(?:www\.)?github\.com/chrislysen/secret-loyalty-probe(?:\.git)?(?:[/?#]|$)"
    r"|^https?://chrislysen\.github\.io/secret-loyalty-probe(?:[/?#]|$)"
    r"|^https?://raw\.githubusercontent\.com/chrislysen/secret-loyalty-probe(?:[/?#]|$)"
)

# Hosts used only to answer "is there a network at all". Probed ONLY when every real URL has already
# failed at the network layer, so the common case costs zero extra requests. github.com is in the
# list deliberately: if we cannot reach GitHub's edge at all we cannot judge our own link either.
CANARIES = ["https://example.com/", "https://www.google.com/generate_204", "https://github.com/"]

UA = "secret-loyalty-probe-link-check/1.0 (+writeup/check_links.py; unauthenticated)"

# Trailing characters markdown glues onto a URL that are not part of it. `)` `]` `>` and the quote
# and backtick and pipe characters are excluded by the pattern itself; these are the ones that can
# legally appear inside a URL and so have to be stripped from the right instead.
_TRAILING = ".,;:!?*_"
URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`|\\]+", re.IGNORECASE)

# A scheme is how a URL is written, not whether a reader can follow it. The first version of this
# gate matched `https?://` only, and so checked ONE of the six external references in the paper --
# missing every `arXiv:` identifier and, worse, the bare `github.com/...` citation that a whole
# retraction rests on. A gate that reports "all 1 of our own URL(s) return 200" while five
# references go unlooked-at is the kind of reassuring number this paper exists to complain about.
BARE_HOST_RE = re.compile(r"(?<![/\w.@:])((?:github\.com|huggingface\.co)/[A-Za-z0-9_.\-]+"
                          r"(?:/[A-Za-z0-9_.\-]+)*)", re.IGNORECASE)
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)


def expand_bare_references(text):
    """Yield (url, as_written) for references written without a scheme.

    arXiv identifiers resolve to their abs/ page. These are reported like any other external
    reference: a non-200 on someone else's host is never fatal, so adding them cannot break the
    build -- it can only stop the gate from overstating its own coverage.
    """
    for m in BARE_HOST_RE.finditer(text):
        host_path = m.group(1).rstrip(_TRAILING)
        yield "https://" + host_path, m.group(0)
    for m in ARXIV_RE.finditer(text):
        yield "https://arxiv.org/abs/" + m.group(1), m.group(0)


class Probe:
    """One URL's result. `kind` is the axis that matters: did we reach a server, or not."""

    def __init__(self, url, where, own):
        self.url = url
        self.where = where          # list of "FILE:LINE" strings
        self.own = own
        self.kind = "unprobed"      # "http" | "network" | "dns" | "malformed"
        self.status = None          # int, when kind == "http"
        self.final_url = None       # after redirects
        self.detail = ""

    @property
    def reached(self) -> bool:
        return self.kind == "http"

    @property
    def failed(self) -> bool:
        """Does this break the build?

        A NON-200 only breaks it for our own URLs -- someone else's outage is not our defect. But a
        MALFORMED url is fatal wherever it points, because it is a defect in the document rather
        than in a server: an unparseable link is dead for every reader, and a `user:pass@host` link
        is a credential published in a PDF. The first version of this gate scoped both to `own`
        only, and a test with `https://tok:x@github.com/...` sailed through as "external" -- the
        credential form does not match a host-anchored own-repo pattern, which is exactly when you
        least want the check to shrug.
        """
        if self.kind == "malformed":
            return True
        if not self.own:
            return False
        return not (self.kind == "http" and self.status == 200)

    def label(self) -> str:
        if self.kind == "http":
            return f"{self.status}"
        if self.kind == "dns":
            return "DNS"
        if self.kind == "network":
            return "NET"
        if self.kind == "malformed":
            return "BAD"
        return "----"


def extract_urls(text: str) -> list[tuple[int, str]]:
    """Every reference a reader could follow, with the 1-based line it sits on, in document order.

    Includes scheme-less `github.com/...` and `arXiv:NNNN.NNNNN` references, which are links to a
    reader even though they are not URLs to a regex.
    """
    found = []
    for lineno, line in enumerate(text.split("\n"), 1):
        explicit = []
        for raw in URL_RE.findall(line):
            url = raw.rstrip(_TRAILING)
            if url:
                explicit.append(url)
                found.append((lineno, url))
        # Do not double-count a bare host that is already inside an explicit URL on the same line.
        for url, as_written in expand_bare_references(line):
            if not any(as_written in e for e in explicit):
                found.append((lineno, url))
    return found


def collect(docs=DOCS) -> tuple[list[Probe], list[str]]:
    """Read the documents; return deduped probes in first-appearance order, plus files missing."""
    own_re = re.compile(OWN_REPO_URL_RE, re.IGNORECASE)
    order: list[str] = []
    seen: dict[str, Probe] = {}
    missing: list[str] = []
    for name in docs:
        path = HERE / name
        if not path.exists():
            missing.append(name)
            continue
        for lineno, url in extract_urls(path.read_text(encoding="utf-8")):
            if url not in seen:
                seen[url] = Probe(url, [], bool(own_re.search(url)))
                order.append(url)
            seen[url].where.append(f"{name}:{lineno}")
    return [seen[u] for u in order], missing


def _opener() -> urllib.request.OpenerDirector:
    """An opener that cannot authenticate as us, and is checked to make sure of it."""
    op = urllib.request.build_opener(
        urllib.request.ProxyHandler(),                                 # honours *_proxy env vars
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        urllib.request.HTTPRedirectHandler(),
    )
    # urllib installs no auth handler unless asked, but "unauthenticated" is this file's entire
    # reason to exist, so it is asserted rather than assumed.
    banned = (urllib.request.HTTPBasicAuthHandler, urllib.request.HTTPDigestAuthHandler,
              urllib.request.HTTPPasswordMgr)
    assert not any(isinstance(h, banned) for h in op.handlers), "link check must carry no credentials"
    op.addheaders = [("User-Agent", UA), ("Accept", "*/*")]
    return op


def _request(url: str, method: str, timeout: float, op) -> tuple[str, int | None, str | None, str]:
    """-> (kind, status, final_url, detail). Never raises."""
    req = urllib.request.Request(url, method=method)
    try:
        with op.open(req, timeout=timeout) as resp:
            return "http", int(resp.status), resp.url, ""
    except urllib.error.HTTPError as e:
        # A server answered. This is a REACHED result even though it is an error status -- and it is
        # the case the whole gate exists for: a private repo answers 404 to a stranger.
        return "http", int(e.code), getattr(e, "url", url), (e.reason or "")
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, socket.gaierror):
            return "dns", None, None, f"name resolution failed ({reason})"
        return "network", None, None, f"{type(reason).__name__}: {reason}"
    except (TimeoutError, socket.timeout):
        return "network", None, None, f"timed out after {timeout:.0f}s"
    except (ConnectionError, ssl.SSLError, OSError) as e:
        return "network", None, None, f"{type(e).__name__}: {e}"
    except ValueError as e:
        return "malformed", None, None, str(e)


def probe(p: Probe, timeout: float, op) -> Probe:
    parts = urlsplit(p.url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        p.kind, p.detail = "malformed", "not an absolute http(s) URL"
        return p
    if "@" in parts.netloc:
        # user:pass@host in a published URL is a credential leak AND makes the check meaningless.
        p.kind, p.detail = "malformed", "URL carries embedded credentials"
        return p

    kind, status, final, detail = _request(p.url, "HEAD", timeout, op)
    # Some servers refuse HEAD outright (405/501) or gate it behind a bot rule (403). Those are
    # about the METHOD, not about reachability, so re-ask with GET before believing them.
    if kind == "http" and status in (403, 405, 501):
        k2, s2, f2, d2 = _request(p.url, "GET", timeout, op)
        if k2 == "http":
            kind, status, final, detail = k2, s2, f2, d2 + " (GET; HEAD was refused)"
    p.kind, p.status, p.final_url, p.detail = kind, status, final, detail
    return p


def network_is_up(timeout: float, op) -> bool:
    for url in CANARIES:
        kind, _, _, _ = _request(url, "HEAD", min(timeout, 6.0), op)
        if kind == "http":
            return True
    return False


def run_check(docs=DOCS, timeout: float | None = None, quiet: bool = False) -> int:
    """The gate. Returns a process exit code; prints the full report either way."""
    say = (lambda *a: None) if quiet else (lambda *a: print(*a))

    if os.environ.get("SKIP_LINK_CHECK") == "1":
        print("=" * 78)
        print("  WARNING: LINK CHECK SKIPPED (SKIP_LINK_CHECK=1)")
        print("  The paper's links are NOT verified in this build. The submitted version of this")
        print("  paper shipped a 404 on its only link; that is what this gate exists to prevent.")
        print("  Unset SKIP_LINK_CHECK and rebuild before publishing or submitting anything.")
        print("=" * 78)
        return 0

    timeout = timeout if timeout is not None else float(os.environ.get("LINK_CHECK_TIMEOUT", "10"))
    probes, missing = collect(docs)
    for name in missing:
        say(f"[links] note: {name} not present, skipped")
    if not probes:
        say("[links] no http(s) URLs found in " + ", ".join(d for d in docs if d not in missing))
        return 0

    say(f"[links] checking {len(probes)} URL(s) UNAUTHENTICATED "
        f"(no token, no netrc, no git config, no gh auth), timeout {timeout:.0f}s")
    op = _opener()
    for p in probes:
        probe(p, timeout, op)

    # Offline is a property of the run, not of the paper. Only claim it when NOTHING answered and an
    # independent canary agrees -- otherwise a single dead host would silence the whole gate.
    offline = False
    if not any(p.reached for p in probes):
        offline = not network_is_up(timeout, op)

    width = max(len(p.url) for p in probes)
    say("")
    say(f"  {'STATUS':>6}  {'SCOPE':<8}  {'URL':<{width}}  WHERE")
    say(f"  {'-'*6}  {'-'*8}  {'-'*width}  {'-'*5}")
    for p in probes:
        scope = "OURS" if p.own else "external"
        say(f"  {p.label():>6}  {scope:<8}  {p.url:<{width}}  {', '.join(p.where)}")
        if p.final_url and p.final_url.rstrip("/") != p.url.rstrip("/"):
            say(f"  {'':>6}  {'':<8}  -> redirected to {p.final_url}")
        if p.detail:
            say(f"  {'':>6}  {'':<8}  -- {p.detail}")
    say("")

    if offline:
        print("=" * 78)
        print("  WARNING: NO NETWORK -- link reachability was NOT verified.")
        print(f"  Every URL and all {len(CANARIES)} canary hosts failed at the network layer (DNS /")
        print("  timeout / refused), so this is an offline build, not a broken link. No URL here")
        print("  returned 404: none of them returned anything at all. Re-run this gate on a")
        print("  networked machine before submitting.")
        print("=" * 78)
        return 0

    bad_url = [p for p in probes if p.kind == "malformed"]
    unreachable = [p for p in probes if p.own and p.kind != "malformed"
                   and not (p.kind == "http" and p.status == 200)]
    ext_bad = [p for p in probes if not p.own and p.kind != "malformed"
               and not (p.kind == "http" and p.status == 200)]
    if ext_bad:
        say(f"[links] {len(ext_bad)} external URL(s) did not return 200 "
            f"(reported, not fatal -- we do not control them)")

    n_own = sum(1 for p in probes if p.own)
    if bad_url:
        print(f"ERROR: {len(bad_url)} URL(s) in the paper are malformed or carry credentials, "
              f"which is a defect in the document and not in any server:", file=sys.stderr)
        for p in bad_url:
            print(f"  {p.label():>6}  {p.url}   [{', '.join(p.where)}]  {p.detail}", file=sys.stderr)
    if unreachable:
        print(f"ERROR: {len(unreachable)} of {n_own} URL(s) pointing at our OWN repository did not "
              f"return 200 to an unauthenticated client:", file=sys.stderr)
        for p in unreachable:
            print(f"  {p.label():>6}  {p.url}   [{', '.join(p.where)}]  {p.detail}", file=sys.stderr)
        print("", file=sys.stderr)
        print("A reviewer following this link sees exactly what is printed above. If the status is",
              file=sys.stderr)
        print("404 on a repository that exists, the repository is PRIVATE -- make it public, or",
              file=sys.stderr)
        print("change the link to something a stranger can actually open.", file=sys.stderr)
    if bad_url or unreachable:
        return 1

    # State the denominator, which is the whole argument of the paper this gate ships with. An
    # earlier version reported only "all N of our own URL(s) return 200" while silently checking one
    # reference out of six, which reads as coverage and is not.
    n_ext = len(probes) - n_own
    n_ext_ok = sum(1 for p in probes if not p.own and p.status == 200)
    if n_own:
        say(f"[links] OK: all {n_own} of our own URL(s) return 200 unauthenticated; "
            f"{n_ext_ok} of {n_ext} external reference(s) also resolve "
            f"(non-200 on someone else's host is reported, never fatal)")
    else:
        say("[links] OK: no URL in these documents points at our own repository "
            "(nothing for a reviewer to find, which may itself be the problem); "
            f"{n_ext_ok} of {n_ext} external reference(s) resolve")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    return run_check()


if __name__ == "__main__":
    raise SystemExit(main())
