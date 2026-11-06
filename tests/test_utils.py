from __future__ import annotations

import codecs
import logging

import pytest

from charset_normalizer.constant import IANA_SUPPORTED
from charset_normalizer.utils import (
    cp_similarity,
    cut_sequence_chunks,
    iana_name,
    is_accentuated,
    set_logging_handler,
)


@pytest.mark.parametrize(
    "character, expected_is_accentuated",
    [
        ("é", True),
        ("è", True),
        ("à", True),
        ("À", True),
        ("Ù", True),
        ("ç", True),
        ("a", False),
        ("€", False),
        ("&", False),
        ("Ö", True),
        ("ü", True),
        ("ê", True),
        ("Ñ", True),
        ("Ý", True),
        ("Ω", False),
        ("ø", False),
        ("Ё", False),
    ],
)
def test_is_accentuated(character, expected_is_accentuated):
    assert (
        is_accentuated(character) is expected_is_accentuated
    ), "is_accentuated behavior incomplete"


@pytest.mark.parametrize(
    "cp_name_a, cp_name_b, expected_is_similar",
    [
        ("cp1026", "cp1140", True),
        ("cp1140", "cp1026", True),
        ("latin_1", "cp1252", True),
        ("latin_1", "iso8859_4", True),
        ("latin_1", "cp1251", False),
        ("cp1251", "mac_turkish", False),
    ],
)
def test_cp_similarity(cp_name_a, cp_name_b, expected_is_similar):
    is_similar = cp_similarity(cp_name_a, cp_name_b) >= 0.8

    assert is_similar is expected_is_similar, "cp_similarity is broken"


@pytest.mark.parametrize("cp_name", IANA_SUPPORTED)
def test_iana_name_resolves_every_supported_encoding(cp_name):
    assert iana_name(cp_name) == cp_name, "IANA_SUPPORTED entry is unresolvable"


@pytest.mark.parametrize("cp_name", IANA_SUPPORTED)
def test_every_supported_encoding_decodes_with_errors_ignore(cp_name):
    # Detection decodes with errors="ignore", which utf_7_imap (Python 3.16+) rejects.
    decoder = codecs.getincrementaldecoder(cp_name)(errors="ignore")
    assert isinstance(decoder.decode(b"@"), str)


def test_cut_sequence_chunks_bad_cutting_repair_does_not_wrap_negative():
    # The multi-byte "bad cutting" repair loop backs up from the current
    # offset looking for a chunk boundary that actually lines up with
    # decoded_payload. When the current offset is small (1, 2 or 3 bytes
    # past a BOM, which is itself only a couple of bytes), backing up 4
    # steps went negative and `sequences[j:chunk_end]` silently wrapped
    # around to slice from the END of the buffer instead of clamping to
    # the start, as the "back up toward the start of the buffer" intent
    # describes.
    sequences = "HELLO".encode("utf_16")  # 2-byte BOM + 10 bytes payload
    assert len(sequences) == 12

    chunk = list(
        cut_sequence_chunks(
            sequences,
            "utf_16",
            range(2, 3),  # i == 2, right after the BOM
            10,  # chunk_size, so chunk_end == i + chunk_size == len(sequences)
            True,  # bom_or_sig_available
            True,  # strip_sig_or_bom
            sequences[:2],  # sig_payload
            True,  # is_multi_byte_decoder
            "ZZZZZZZZZZ",  # decoded_payload: deliberately won't match any
            # correctly-bounded candidate chunk, forcing the repair loop
            # to exhaust every non-negative j.
        )
    )[0]

    # With the wraparound bug, j reaches -1, sequences[-1:12] slices the
    # single trailing byte at the end of the buffer, decodes (with
    # errors="ignore") to "", and "" is trivially "found" in any string,
    # so the loop breaks early on that bogus empty chunk. The fix clamps
    # j at 0, so the loop instead exhausts on the correctly-bounded
    # candidate from j == 0 (the original cut, BOM included), which
    # decodes to the real payload.
    assert chunk == "HELLO"
