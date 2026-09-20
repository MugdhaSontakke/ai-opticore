"""Security helpers: URL/SSRF validation and secret redaction.

Policy goals:
- Provider endpoints (``base_url``) are user-controlled input. We validate
  scheme, reject credentials embedded in the URL, and refuse connections to
  network ranges that are not meant to be exposed when the caller opts in to
  the stricter policy.
- Local Ollama/vLLM/llama.cpp servers run on loopback or private ranges by
  design, so private/loopback networks are ALLOWED by default and can be
  tightened via ``allow_private_networks`` / ``allowed_hosts``.
- Exception messages are never allowed to leak API keys; every provider
  sanitizes them with :func:`redact_text` before raising.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Iterable
from urllib.parse import urlparse

from opticore.exceptions import ConfigurationError

logger = logging.getLogger("opticore.security")

_ALLOWED_SCHEMES = ("http", "https")

# Metadata endpoint (cloud IMDS) and other ranges that should never be
# reachable from a generic HTTP client unless explicitly allowed.
# 169.254.169.254 is the classic SSRF target (AWS/GCP/Azure instance
# metadata). 100.100.100.200 is Alibaba Cloud's metadata endpoint.
_METADATA_IPS = (
    "169.254.169.254",
    "100.100.100.200",
    "fd00:ec2::254",
)

_SPECIAL_LOOPBACK = (
    "127.0.0.0/8",  # IPv4 loopback
    "::1",          # IPv6 loopback
    "::ffff:127.0.0.0/104",
)

_PRIVATE = (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "fc00::/7",     # IPv6 unique local
    "fe80::/10",    # IPv6 link-local
)

_RESERVED_MORE = (
    "0.0.0.0/8",
    "100.64.0.0/10",  # CGNAT
    "192.0.0.0/24",
    "192.0.2.0/24",   # TEST-NET
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",    # multicast
    "240.0.0.0/4",    # reserved
    "::/128",
    "::1",
    "::ffff:169.254.169.254",
)


def _build_networks(specs: tuple[str, ...]) -> list[ipaddress._BaseNetwork]:
    out: list[ipaddress._BaseNetwork] = []
    for spec in specs:
        try:
            out.append(ipaddress.ip_network(spec))
        except ValueError:  # pragma: no cover - literals above are valid
            continue
    return out


_LOOPBACK_NETWORKS = _build_networks(_SPECIAL_LOOPBACK)
_BLOCKED_WHEN_PRIVATE = _build_networks((*_PRIVATE, *_RESERVED_MORE))


def _ip_in_networks(addr: ipaddress._BaseAddress, nets: Iterable[ipaddress._BaseNetwork]) -> bool:
    return any(addr.version == net.version and addr in net for net in nets)


def _is_metadata(addr: ipaddress._BaseAddress) -> bool:
    for candidate in _METADATA_IPS:
        try:
            target = ipaddress.ip_address(candidate)
        except ValueError:  # pragma: no cover
            continue
        if target == addr:
            return True
    return False


def _resolve_host(host: str) -> list[ipaddress._BaseAddress]:
    """Resolve a hostname to its IP addresses (literal IPs pass through)."""
    try:
        info = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ConfigurationError(
            f"Cannot validate endpoint host {host!r}: name resolution failed "
            f"({exc}). Fix the URL or add the host to 'allowed_hosts'."
        ) from exc
    seen: list[ipaddress._BaseAddress] = []
    for entry in info:
        try:
            addr = ipaddress.ip_address(str(entry[4][0]).split("%")[0])
        except ValueError:
            continue
        if addr not in seen:
            seen.append(addr)
    return seen or []


def validate_base_url(
    url: str,
    *,
    allow_private_networks: bool = True,
    allowed_hosts: list[str] | None = None,
    allow_loopback: bool = True,
) -> str:
    """Validate a provider ``base_url`` and return the normalized URL.

    Raises :class:`ConfigurationError` when the URL uses an unsupported
    scheme, embeds credentials, targets a blocked network, or cannot be
    resolved. Returns the URL (trailing slash stripped) on success.

    ``allowed_hosts`` always overrides network checks (explicit allow-list).
    ``allow_private_networks=False`` blocks RFC1918/ULA/CGNAT ranges (and
    metadata endpoints) -- use this only when your providers are all
    public/global.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise ConfigurationError("base_url must be a non-empty http(s) URL")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ConfigurationError(
            f"base_url scheme must be http or https, got {parsed.scheme!r}. "
            "Other schemes are not supported for provider endpoints."
        )
    if not parsed.hostname:
        raise ConfigurationError(
            f"base_url has no host: {url!r}. Provide a full http(s) URL."
        )
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError(
            "base_url must not embed credentials (user:password@host). "
            "Authenticate via environment variables instead."
        )

    host = parsed.hostname
    if allowed_hosts:
        lowered = [h.lower().rstrip(".") for h in allowed_hosts]
        if host.lower().rstrip(".") in lowered:
            return url.rstrip("/")

    allowlist_hit = False
    if allow_loopback and host in ("localhost", "localhost.localdomain"):
        allowlist_hit = True
    addrs = _resolve_host(host) if not allowlist_hit else [ipaddress.ip_address("127.0.0.1")]
    if not addrs:
        raise ConfigurationError(f"Cannot resolve endpoint host {host!r}")

    def _blocked(addr: ipaddress._BaseAddress) -> str | None:
        if _is_metadata(addr):
            return "metadata endpoint (SSRF target)"
        if not allow_private_networks and _ip_in_networks(addr, _BLOCKED_WHEN_PRIVATE):
            return "private/reserved network (blocked by allow_private_networks=False)"
        if not allow_loopback and _ip_in_networks(addr, _LOOPBACK_NETWORKS):
            return "loopback address (blocked by allow_loopback=False)"
        return None

    blocked: list[str] = []
    for addr in addrs:
        reason = _blocked(addr)
        if reason:
            blocked.append(f"{host}={addr} ({reason})")
    if blocked:
        raise ConfigurationError(
            "base_url resolves to a network that is blocked: " + "; ".join(blocked) +
            ". If you intend to reach a local/private LLM server, keep "
            "'allow_private_networks' enabled (the default) or add the host "
            "to 'allowed_hosts'."
        )
    return url.rstrip("/")


def redact_text(message: str) -> str:
    """Redact likely secrets from an arbitrary string (e.g. an error message).

    Reuses the same rules as the logging :class:`RedactingFilter` so raised
    exceptions can never leak API keys into logs, dashboards, or tests.
    """
    from opticore.logging import redact_text as _log_redact

    return _log_redact(message)
