from os import PathLike
from typing import List, BinaryIO, Optional, Set

from charset_normalizer.constant import TOO_SMALL_SEQUENCE, TOO_BIG_SEQUENCE, IANA_SUPPORTED
from charset_normalizer.md import mess_ratio
from charset_normalizer.models import CharsetMatches, CharsetMatch
from warnings import warn
import logging

from charset_normalizer.utils import any_specified_encoding, is_multi_byte_encoding, identify_sig_or_bom, range_scan, \
    should_strip_sig_or_bom, is_cp_similar
from charset_normalizer.cd import coherence_ratio, encoding_languages, mb_encoding_languages, merge_coherence_ratios

logger = logging.getLogger("charset_normalizer")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(handler)


def from_bytes(
        sequences: bytes,
        steps: int = 5,
        chunk_size: int = 512,
        threshold: float = 0.2,
        cp_isolation: List[str] = None,
        cp_exclusion: List[str] = None,
        preemptive_behaviour: bool = True,
        explain: bool = False
) -> CharsetMatches:

    if not explain:
        logger.setLevel(logging.ERROR)

    if cp_isolation is not None:
        logger.warning('cp_isolation is set. use this flag for debugging purpose. '
                       'limited list of encoding allowed : %s.',
                       ', '.join(cp_isolation))
    else:
        cp_isolation = []

    if cp_exclusion is not None:
        logger.warning(
            'cp_exclusion is set. use this flag for debugging purpose. '
            'limited list of encoding excluded : %s.',
            ', '.join(cp_exclusion))
    else:
        cp_exclusion = []

    length = len(sequences)  # type: int

    if length == 0:
        return CharsetMatches(
            [
                CharsetMatch(
                    sequences,
                    "utf_8",
                    0.,
                    False,
                    [],
                    [],
                    ""
                )
            ]
        )

    if length <= (chunk_size * steps):
        logger.warning(
            'override steps (%i) and chunk_size (%i) as content does not fit (%i byte(s) given) parameters.',
            steps, chunk_size, length)
        steps = 1
        chunk_size = length

    if steps > 1 and length / steps < chunk_size:
        chunk_size = int(length / steps)

    is_too_small_sequence = len(sequences) < TOO_SMALL_SEQUENCE  # type: bool
    is_too_large_sequence = len(sequences) >= TOO_BIG_SEQUENCE  # type: bool

    if is_too_small_sequence:
        warn(f'Trying to detect encoding from a tiny portion of ({length}) byte(s).')

    prioritized_encodings = []  # type: List[str]

    specified_encoding = any_specified_encoding(sequences) if preemptive_behaviour is True else None  # type: Optional[str]

    if specified_encoding is not None:
        prioritized_encodings.append(specified_encoding)
        logger.info('Detected declarative mark in sequence. Priority +1 given for %s.', specified_encoding)

    tested = set()  # type: Set[str]
    tested_but_hard_failure = []  # type: List[str]
    tested_but_soft_failure = []  # type: List[str]

    single_byte_hard_failure_count = 0  # type: int
    single_byte_soft_failure_count = 0  # type: int

    results = CharsetMatches()  # type: CharsetMatches

    sig_encoding, sig_payload = identify_sig_or_bom(sequences)

    if sig_encoding is not None:
        prioritized_encodings.append(sig_encoding)
        logger.info('Detected a SIG or BOM mark on first %i byte(s). Priority +1 given for %s.', len(sig_payload), sig_encoding)

    prioritized_encodings.append("ascii")

    if "utf_8" not in prioritized_encodings:
        prioritized_encodings.append("utf_8")

    for encoding_iana in prioritized_encodings+IANA_SUPPORTED:

        if cp_isolation and encoding_iana not in cp_isolation:
            continue

        if cp_exclusion and encoding_iana in cp_exclusion:
            continue

        if encoding_iana in tested:
            continue

        tested.add(encoding_iana)

        decoded_payload = None  # type: Optional[str]
        bom_or_sig_available = sig_encoding == encoding_iana  # type: bool
        strip_sig_or_bom = bom_or_sig_available and should_strip_sig_or_bom(encoding_iana)  # type: bool

        if encoding_iana in {"utf_16", "utf_32"} and bom_or_sig_available is False:
            logger.info("Encoding %s wont be tested as-is because it require a BOM. Will try some sub-encoder LE/BE.", encoding_iana)
            continue

        try:
            is_multi_byte_decoder = is_multi_byte_encoding(encoding_iana)  # type: bool
        except (ModuleNotFoundError, ImportError):
            logger.debug("Encoding %s does not provide an IncrementalDecoder", encoding_iana)
            continue

        try:
            if is_too_large_sequence and is_multi_byte_decoder is False:
                str(
                    sequences[:int(50e4)],
                    encoding=encoding_iana
                )
            else:
                decoded_payload = str(
                    sequences if strip_sig_or_bom is False else sequences[len(sig_payload):],
                    encoding=encoding_iana
                )
        except UnicodeDecodeError as e:
            logger.debug('Code page %s does not fit given bytes sequence at ALL. %s', encoding_iana, str(e))
            tested_but_hard_failure.append(encoding_iana)
            if not is_multi_byte_decoder:
                single_byte_hard_failure_count += 1
            continue
        except LookupError:
            tested_but_hard_failure.append(encoding_iana)
            if not is_multi_byte_decoder:
                single_byte_hard_failure_count += 1
            continue

        similar_soft_failure_test = False  # type: bool

        for encoding_soft_failed in tested_but_soft_failure:
            if is_cp_similar(encoding_iana, encoding_soft_failed):
                similar_soft_failure_test = True
                break

        if similar_soft_failure_test:
            logger.warning("%s is deemed too similar to code page %s and was consider unsuited already. Continuing!", encoding_iana, encoding_soft_failed)
            continue

        r_ = range(
            0 if strip_sig_or_bom is False else len(sig_payload) + 1,
            length,
            int(length / steps)
        )

        multi_byte_bonus = is_multi_byte_decoder and len(decoded_payload) < length  # type: bool

        if multi_byte_bonus:
            logger.info('Code page %s is a multi byte encoding table and it appear that at least one character was encoded using n-bytes. Should not be a coincidence. Priority +1 given.', encoding_iana)

        max_chunk_gave_up = int(len(r_) / 4)  # type: int

        if max_chunk_gave_up < 2:
            max_chunk_gave_up = 2

        early_stop_count = 0  # type: int

        md_chunks: List[str] = []
        md_ratios = []

        for i in r_:
            chunk: str = sequences[i:i + chunk_size].decode(encoding_iana, errors="ignore")

            md_chunks.append(chunk)

            md_ratios.append(
                mess_ratio(
                    chunk,
                    threshold
                )
            )

            if md_ratios[-1] >= threshold:
                early_stop_count += 1

            if early_stop_count >= max_chunk_gave_up:
                break

        mean_mess_ratio = sum(md_ratios) / len(md_ratios)  # type: float

        if mean_mess_ratio >= threshold or early_stop_count >= max_chunk_gave_up:
            tested_but_soft_failure.append(encoding_iana)
            if not is_multi_byte_decoder:
                single_byte_soft_failure_count += 1
            logger.warning('%s was excluded because of initial chaos probing. Gave up %i time(s). '
                           'Computed mean chaos is %f %%.',
                           encoding_iana,
                           early_stop_count,
                           round(mean_mess_ratio * 100, ndigits=3))
            continue

        logger.info(
            '%s passed initial chaos probing. Mean measured chaos is %f %%',
            encoding_iana,
            round(mean_mess_ratio * 100, ndigits=3)
        )

        if not is_multi_byte_decoder:
            target_languages = encoding_languages(encoding_iana)  # type: List[str]
        else:
            target_languages = mb_encoding_languages(encoding_iana)  # type: List[str]

        if target_languages:
            logger.debug(f"{encoding_iana} should target any language(s) of {str(target_languages)}")

        cd_ratios = []

        for chunk in md_chunks:
            chunk_languages = coherence_ratio(chunk, 0.1, ",".join(target_languages) if target_languages else None)

            cd_ratios.append(
                chunk_languages
            )

        cd_ratios_merged = merge_coherence_ratios(cd_ratios)

        if cd_ratios_merged:
            logger.debug(f"We detected language {cd_ratios_merged} using {encoding_iana}")

        results.add(
            CharsetMatch(
                sequences,
                encoding_iana,
                mean_mess_ratio,
                bom_or_sig_available,
                cd_ratios_merged,
                [],
                decoded_payload
            )
        )

        if encoding_iana in [specified_encoding, "ascii", "utf_8"] and mean_mess_ratio < 0.1:
            logger.info("%s is most likely the one. Stopping the process.", encoding_iana)
            return CharsetMatches(
                [results[encoding_iana]]
            )

        if encoding_iana == sig_encoding:
            logger.info(
                "%s is most likely the one as we detected a BOM or SIG within the beginning of the sequence.",
                encoding_iana
            )
            return CharsetMatches(
                [results[encoding_iana]]
            )

        if results[-1].languages:
            logger.info(
                "Using %s code page we detected the following languages: %s",
                encoding_iana,
                results[-1]._languages
            )

    return results


def from_fp(
        fp: BinaryIO,
        steps: int = 5,
        chunk_size: int = 512,
        threshold: float = 0.20,
        cp_isolation: List[str] = None,
        cp_exclusion: List[str] = None,
        preemptive_behaviour: bool = True,
        explain: bool = False
) -> CharsetMatches:
    return from_bytes(
        fp.read(),
        steps,
        chunk_size,
        threshold,
        cp_isolation,
        cp_exclusion,
        preemptive_behaviour,
        explain
    )


def from_path(
        path: PathLike,
        steps: int = 5,
        chunk_size: int = 512,
        threshold: float = 0.20,
        cp_isolation: List[str] = None,
        cp_exclusion: List[str] = None,
        preemptive_behaviour: bool = True,
        explain: bool = False
) -> CharsetMatches:
    with open(path, 'rb') as fp:
        return from_fp(fp, steps, chunk_size, threshold, cp_isolation, cp_exclusion, preemptive_behaviour, explain)
