"""Network egress guard — the single chokepoint for the local-only guarantee.

When ``[runtime] local_only`` is on, OpenJarvis must run 100% on the local
machine. This module provides two layers of that guarantee:

1. :class:`LocalOnlyViolation` — a loud, catchable error raised by the engine
   factory when a cloud backend is requested in ``local_only`` mode (see
   ``openjarvis.engine._discovery``). It fails *closed*: never a silent cloud
   fallback.

2. A process-wide **socket egress guard** that blocks every outbound TCP/UDP
   connection except to the allowlist — loopback addresses, the configured
   local engine hosts, and any extra hosts in ``[runtime] egress_allowlist``.
   AF_UNIX (local IPC) is always allowed. This is the chokepoint every tool and
   engine passes through, because they all ultimately call ``socket.connect``.

The guard is the *inverse* of ``openjarvis.security.ssrf`` (which blocks private
IPs to stop SSRF): here loopback/private-local services are exactly what we
allow, and the public internet is what we block. The two are complementary.

Design notes:

* The guard is **opt-in by call**, never installed at import time, so importing
  this module (or ``openjarvis``) never changes socket behaviour. Activate it
  explicitly via :func:`enforce_local_only` / :func:`install_guard`.
* Blocking raises :class:`EgressBlocked` (an ``OSError`` subclass) from inside
  ``connect``/``connect_ex`` so HTTP clients (httpx, requests, urllib) surface
  it as a connection error with a clear, local-only message — and callers that
  ``except OSError`` still get a sane failure rather than a crash.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from typing import TYPE_CHECKING, Iterable, Optional, Set
from urllib.parse import urlparse

if TYPE_CHECKING:
    from openjarvis.core.config import JarvisConfig

__all__ = [
    "EgressBlocked",
    "LocalOnlyViolation",
    "build_allowlist",
    "enforce_local_only",
    "host_allowed",
    "install_guard",
    "is_guard_active",
    "uninstall_guard",
    "url_allowed",
]


class LocalOnlyViolation(RuntimeError):
    """Raised when a cloud-only capability is requested while ``local_only``.

    Distinct from :class:`EgressBlocked`: this is the *policy* failure surfaced
    by the engine factory (a cloud engine was explicitly requested) and is a
    plain ``RuntimeError`` so it is loud and not swallowed by ``except OSError``
    networking handlers.
    """


class EgressBlocked(OSError):
    """Raised by the socket guard when an outbound connection is blocked.

    Subclasses ``OSError`` so it integrates with socket-based clients' error
    handling, but carries a clear local-only message.
    """


# Hostnames that always denote the local machine.
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})

# ---------------------------------------------------------------------------
# Allowlist construction + matching
# ---------------------------------------------------------------------------


def _normalize_host(host: str) -> str:
    """Lower-case, strip brackets/whitespace from a host string."""
    h = host.strip().lower()
    if h.startswith("[") and h.endswith("]"):  # bracketed IPv6 literal
        h = h[1:-1]
    return h


def _host_from_entry(entry: str) -> Optional[str]:
    """Extract a bare host from an allowlist entry (``host`` or ``host:port``).

    Bare IPv6 literals (which contain colons) are detected and returned as-is;
    only a trailing ``:port`` on a non-IPv6 entry is stripped.
    """
    e = entry.strip()
    if not e:
        return None
    # Strip a URL scheme if the user pasted one (http://host:port).
    if "://" in e:
        e = urlparse(e).hostname or ""
        return _normalize_host(e) if e else None
    e = _normalize_host(e)
    # IPv6 literal? (more than one colon and parses as an address)
    if e.count(":") >= 2:
        try:
            ipaddress.ip_address(e)
            return e
        except ValueError:
            pass
    # host:port → host
    if ":" in e:
        e = e.rsplit(":", 1)[0]
    return e or None


def build_allowlist(config: Optional["JarvisConfig"] = None) -> Set[str]:
    """Build the set of allowed hosts for the egress guard.

    Always includes loopback names. When *config* is given, adds the configured
    local engine hosts (``engine.*_host`` / per-engine ``host``) and any
    explicit ``runtime.egress_allowlist`` entries. Returned hosts are
    normalised (lower-cased, de-bracketed); IP-literal matching is handled
    separately by :func:`host_allowed`.
    """
    allow: Set[str] = set(_LOOPBACK_NAMES)

    if config is None:
        return allow

    # Explicit user allowlist entries.
    extra = getattr(getattr(config, "runtime", None), "egress_allowlist", "") or ""
    for entry in extra.split(","):
        host = _host_from_entry(entry)
        if host:
            allow.add(host)

    # Configured local engine hosts. EngineConfig holds both ``<engine>_host``
    # string fields and nested per-engine sub-configs that may carry ``host``.
    engine_cfg = getattr(config, "engine", None)
    if engine_cfg is not None:
        for attr in dir(engine_cfg):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(engine_cfg, attr)
            except Exception:
                continue
            if isinstance(value, str) and ("://" in value or attr.endswith("host")):
                host = _host_from_entry(value)
                if host:
                    allow.add(host)
            else:
                nested_host = getattr(value, "host", None)
                if isinstance(nested_host, str) and nested_host:
                    host = _host_from_entry(nested_host)
                    if host:
                        allow.add(host)
    return allow


def _is_loopback_ip(host: str) -> bool:
    """True when *host* is an IP literal in a loopback range."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    # Map IPv4-in-IPv6 (::ffff:127.0.0.1) down to the embedded IPv4.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback


def host_allowed(host: str, allowlist: Iterable[str]) -> bool:
    """Return True if *host* may be contacted under the current allowlist.

    A host is allowed when it is a loopback IP, a loopback name, an exact
    (normalised) match against the allowlist, or a hostname that resolves
    entirely to loopback addresses.
    """
    if not host:
        return False
    h = _normalize_host(host)
    if h in _LOOPBACK_NAMES:
        return True
    allow = {_normalize_host(a) for a in allowlist}
    if h in allow:
        return True
    if _is_loopback_ip(h):
        return True
    # Hostname (not an IP literal): allow only if every resolved address is
    # loopback. This stops a name that resolves to a public IP from sneaking
    # through while still permitting custom loopback aliases in /etc/hosts.
    try:
        ipaddress.ip_address(h)
        return False  # it *was* an IP literal and already failed the checks
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(h, None)
    except socket.gaierror:
        return False
    resolved = {info[4][0] for info in infos}
    return bool(resolved) and all(_is_loopback_ip(ip) for ip in resolved)


def url_allowed(url: str, allowlist: Iterable[str]) -> bool:
    """Convenience: check a URL's host against the allowlist."""
    host = urlparse(url).hostname
    return host_allowed(host or "", allowlist)


# ---------------------------------------------------------------------------
# Process-wide socket guard
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_active = False
_allowlist: Set[str] = set()
_orig_connect = None
_orig_connect_ex = None


def is_guard_active() -> bool:
    """Return True when the socket egress guard is installed."""
    return _active


def _extract_host(address: object) -> Optional[str]:
    """Pull the host string from a ``socket.connect`` address argument.

    Returns ``None`` for non-IP families (e.g. AF_UNIX paths), which the guard
    always permits.
    """
    if isinstance(address, tuple) and address:
        host = address[0]
        return host if isinstance(host, str) else None
    return None


def _check(address: object) -> None:
    """Raise :class:`EgressBlocked` if *address* is outside the allowlist."""
    host = _extract_host(address)
    if host is None:
        return  # AF_UNIX / unknown family → local IPC, allow
    if not host_allowed(host, _allowlist):
        raise EgressBlocked(
            f"Outbound connection to {host!r} blocked: OpenJarvis is running in "
            f"local_only mode (airgap). Allowed: loopback + configured local "
            f"engines. Set OPENJARVIS_LOCAL_ONLY=0 or add the host to "
            f"[runtime] egress_allowlist to permit it."
        )


def install_guard(allowlist: Iterable[str]) -> None:
    """Install the process-wide socket egress guard (idempotent).

    Patches ``socket.socket.connect`` / ``connect_ex`` so any outbound
    connection to a non-allowlisted host raises :class:`EgressBlocked`.
    """
    global _active, _allowlist, _orig_connect, _orig_connect_ex
    with _lock:
        _allowlist = {_normalize_host(a) for a in allowlist}
        if _active:
            return  # already installed; allowlist refreshed above
        _orig_connect = socket.socket.connect
        _orig_connect_ex = socket.socket.connect_ex

        def _guarded_connect(self, address, *args, **kwargs):  # type: ignore[no-untyped-def]
            _check(address)
            return _orig_connect(self, address, *args, **kwargs)

        def _guarded_connect_ex(self, address, *args, **kwargs):  # type: ignore[no-untyped-def]
            _check(address)
            return _orig_connect_ex(self, address, *args, **kwargs)

        socket.socket.connect = _guarded_connect  # type: ignore[assignment,method-assign]
        socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[assignment,method-assign]
        _active = True


def uninstall_guard() -> None:
    """Remove the socket egress guard, restoring the original methods."""
    global _active, _orig_connect, _orig_connect_ex
    with _lock:
        if not _active:
            return
        if _orig_connect is not None:
            socket.socket.connect = _orig_connect  # type: ignore[assignment,method-assign]
        if _orig_connect_ex is not None:
            socket.socket.connect_ex = _orig_connect_ex  # type: ignore[assignment,method-assign]
        _orig_connect = None
        _orig_connect_ex = None
        _active = False


def enforce_local_only(config: "JarvisConfig") -> bool:
    """Install the egress guard iff the config asks for it.

    Returns True when the guard was (or already is) active afterwards. A no-op
    when ``local_only`` is off or ``enforce_egress_guard`` is false, so it is
    safe to call unconditionally from a CLI/daemon bootstrap.
    """
    runtime = getattr(config, "runtime", None)
    if runtime is None or not getattr(runtime, "local_only", False):
        return False
    if not getattr(runtime, "enforce_egress_guard", True):
        return False
    install_guard(build_allowlist(config))
    return True
