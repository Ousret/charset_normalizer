from codecs import BOM_UTF8, BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE
from typing import Dict, List, Union
from encodings.aliases import aliases
from re import compile, IGNORECASE
from collections import OrderedDict

# Contain for each eligible encoding a list of/item bytes SIG/BOM
ENCODING_MARKS = OrderedDict([
    ('utf_8', BOM_UTF8),
    ('utf_7', [
        b'\x2b\x2f\x76\x38',
        b'\x2b\x2f\x76\x39',
        b'\x2b\x2f\x76\x2b',
        b'\x2b\x2f\x76\x2f',
        b'\x2b\x2f\x76\x38\x2d'
    ]),
    ('gb18030', b'\x84\x31\x95'),
    ('utf_32', [
        BOM_UTF32_BE,
        BOM_UTF32_LE
    ]),
    ('utf_16', [
        BOM_UTF16_BE,
        BOM_UTF16_LE
    ]),
])  # type: Dict[str, Union[bytes, List[bytes]]]

TOO_SMALL_SEQUENCE = 32  # type: int
TOO_BIG_SEQUENCE = int(10e6)  # type: int

UTF8_MAXIMAL_ALLOCATION = 1112064  # type: int

UNICODE_RANGES_COMBINED = {'Control character': range(0, 31+1), 'Basic Latin': range(32, 127+1), 'Latin-1 Supplement': range(128, 255+1), 'Latin Extended-A': range(256, 383+1), 'Latin Extended-B': range(384, 591+1), 'IPA Extensions': range(592, 687+1), 'Spacing Modifier Letters': range(688, 767+1), 'Combining Diacritical Marks': range(768, 879+1), 'Greek and Coptic': range(880, 1023+1), 'Cyrillic': range(1024, 1279+1), 'Cyrillic Supplement': range(1280, 1327+1), 'Armenian': range(1328, 1423+1), 'Hebrew': range(1424, 1535+1), 'Arabic': range(1536, 1791+1), 'Syriac': range(1792, 1871+1), 'Arabic Supplement': range(1872, 1919+1), 'Thaana': range(1920, 1983+1), 'NKo': range(1984, 2047+1), 'Samaritan': range(2048, 2111+1), 'Mandaic': range(2112, 2143+1), 'Syriac Supplement': range(2144, 2159+1), 'Arabic Extended-A': range(2208, 2303+1), 'Devanagari': range(2304, 2431+1), 'Bengali': range(2432, 2559+1), 'Gurmukhi': range(2560, 2687+1), 'Gujarati': range(2688, 2815+1), 'Oriya': range(2816, 2943+1), 'Tamil': range(2944, 3071+1), 'Telugu': range(3072, 3199+1), 'Kannada': range(3200, 3327+1), 'Malayalam': range(3328, 3455+1), 'Sinhala': range(3456, 3583+1), 'Thai': range(3584, 3711+1), 'Lao': range(3712, 3839+1), 'Tibetan': range(3840, 4095+1), 'Myanmar': range(4096, 4255+1), 'Georgian': range(4256, 4351+1), 'Hangul Jamo': range(4352, 4607+1), 'Ethiopic': range(4608, 4991+1), 'Ethiopic Supplement': range(4992, 5023+1), 'Cherokee': range(5024, 5119+1), 'Unified Canadian Aboriginal Syllabics': range(5120, 5759+1), 'Ogham': range(5760, 5791+1), 'Runic': range(5792, 5887+1), 'Tagalog': range(5888, 5919+1), 'Hanunoo': range(5920, 5951+1), 'Buhid': range(5952, 5983+1), 'Tagbanwa': range(5984, 6015+1), 'Khmer': range(6016, 6143+1), 'Mongolian': range(6144, 6319+1), 'Unified Canadian Aboriginal Syllabics Extended': range(6320, 6399+1), 'Limbu': range(6400, 6479+1), 'Tai Le': range(6480, 6527+1), 'New Tai Lue': range(6528, 6623+1), 'Khmer Symbols': range(6624, 6655+1), 'Buginese': range(6656, 6687+1), 'Tai Tham': range(6688, 6831+1), 'Combining Diacritical Marks Extended': range(6832, 6911+1), 'Balinese': range(6912, 7039+1), 'Sundanese': range(7040, 7103+1), 'Batak': range(7104, 7167+1), 'Lepcha': range(7168, 7247+1), 'Ol Chiki': range(7248, 7295+1), 'Cyrillic Extended C': range(7296, 7311+1), 'Sundanese Supplement': range(7360, 7375+1), 'Vedic Extensions': range(7376, 7423+1), 'Phonetic Extensions': range(7424, 7551+1), 'Phonetic Extensions Supplement': range(7552, 7615+1), 'Combining Diacritical Marks Supplement': range(7616, 7679+1), 'Latin Extended Additional': range(7680, 7935+1), 'Greek Extended': range(7936, 8191+1), 'General Punctuation': range(8192, 8303+1), 'Superscripts and Subscripts': range(8304, 8351+1), 'Currency Symbols': range(8352, 8399+1), 'Combining Diacritical Marks for Symbols': range(8400, 8447+1), 'Letterlike Symbols': range(8448, 8527+1), 'Number Forms': range(8528, 8591+1), 'Arrows': range(8592, 8703+1), 'Mathematical Operators': range(8704, 8959+1), 'Miscellaneous Technical': range(8960, 9215+1), 'Control Pictures': range(9216, 9279+1), 'Optical Character Recognition': range(9280, 9311+1), 'Enclosed Alphanumerics': range(9312, 9471+1), 'Box Drawing': range(9472, 9599+1), 'Block Elements': range(9600, 9631+1), 'Geometric Shapes': range(9632, 9727+1), 'Miscellaneous Symbols': range(9728, 9983+1), 'Dingbats': range(9984, 10175+1), 'Miscellaneous Mathematical Symbols-A': range(10176, 10223+1), 'Supplemental Arrows-A': range(10224, 10239+1), 'Braille Patterns': range(10240, 10495+1), 'Supplemental Arrows-B': range(10496, 10623+1), 'Miscellaneous Mathematical Symbols-B': range(10624, 10751+1), 'Supplemental Mathematical Operators': range(10752, 11007+1), 'Miscellaneous Symbols and Arrows': range(11008, 11263+1), 'Glagolitic': range(11264, 11359+1), 'Latin Extended-C': range(11360, 11391+1), 'Coptic': range(11392, 11519+1), 'Georgian Supplement': range(11520, 11567+1), 'Tifinagh': range(11568, 11647+1), 'Ethiopic Extended': range(11648, 11743+1), 'Cyrillic Extended-A': range(11744, 11775+1), 'Supplemental Punctuation': range(11776, 11903+1), 'CJK Radicals Supplement': range(11904, 12031+1), 'Kangxi Radicals': range(12032, 12255+1), 'Ideographic Description Characters': range(12272, 12287+1), 'CJK Symbols and Punctuation': range(12288, 12351+1), 'Hiragana': range(12352, 12447+1), 'Katakana': range(12448, 12543+1), 'Bopomofo': range(12544, 12591+1), 'Hangul Compatibility Jamo': range(12592, 12687+1), 'Kanbun': range(12688, 12703+1), 'Bopomofo Extended': range(12704, 12735+1), 'CJK Strokes': range(12736, 12783+1), 'Katakana Phonetic Extensions': range(12784, 12799+1), 'Enclosed CJK Letters and Months': range(12800, 13055+1), 'CJK Compatibility': range(13056, 13311+1), 'CJK Unified Ideographs Extension A': range(13312, 19903+1), 'Yijing Hexagram Symbols': range(19904, 19967+1), 'CJK Unified Ideographs': range(19968, 40959+1), 'Yi Syllables': range(40960, 42127+1), 'Yi Radicals': range(42128, 42191+1), 'Lisu': range(42192, 42239+1), 'Vai': range(42240, 42559+1), 'Cyrillic Extended-B': range(42560, 42655+1), 'Bamum': range(42656, 42751+1), 'Modifier Tone Letters': range(42752, 42783+1), 'Latin Extended-D': range(42784, 43007+1), 'Syloti Nagri': range(43008, 43055+1), 'Common Indic Number Forms': range(43056, 43071+1), 'Phags-pa': range(43072, 43135+1), 'Saurashtra': range(43136, 43231+1), 'Devanagari Extended': range(43232, 43263+1), 'Kayah Li': range(43264, 43311+1), 'Rejang': range(43312, 43359+1), 'Hangul Jamo Extended-A': range(43360, 43391+1), 'Javanese': range(43392, 43487+1), 'Myanmar Extended-B': range(43488, 43519+1), 'Cham': range(43520, 43615+1), 'Myanmar Extended-A': range(43616, 43647+1), 'Tai Viet': range(43648, 43743+1), 'Meetei Mayek Extensions': range(43744, 43775+1), 'Ethiopic Extended-A': range(43776, 43823+1), 'Latin Extended-E': range(43824, 43887+1), 'Cherokee Supplement': range(43888, 43967+1), 'Meetei Mayek': range(43968, 44031+1), 'Hangul Syllables': range(44032, 55215+1), 'Hangul Jamo Extended-B': range(55216, 55295+1), 'High Surrogates': range(55296, 56191+1), 'High Private Use Surrogates': range(56192, 56319+1), 'Low Surrogates': range(56320, 57343+1), 'Private Use Area': range(57344, 63743+1), 'CJK Compatibility Ideographs': range(63744, 64255+1), 'Alphabetic Presentation Forms': range(64256, 64335+1), 'Arabic Presentation Forms-A': range(64336, 65023+1), 'Variation Selectors': range(65024, 65039+1), 'Vertical Forms': range(65040, 65055+1), 'Combining Half Marks': range(65056, 65071+1), 'CJK Compatibility Forms': range(65072, 65103+1), 'Small Form Variants': range(65104, 65135+1), 'Arabic Presentation Forms-B': range(65136, 65279+1), 'Halfwidth and Fullwidth Forms': range(65280, 65519+1), 'Specials': range(65520, 65535+1), 'Linear B Syllabary': range(65536, 65663+1), 'Linear B Ideograms': range(65664, 65791+1), 'Aegean Numbers': range(65792, 65855+1), 'Ancient Greek Numbers': range(65856, 65935+1), 'Ancient Symbols': range(65936, 65999+1), 'Phaistos Disc': range(66000, 66047+1), 'Lycian': range(66176, 66207+1), 'Carian': range(66208, 66271+1), 'Coptic Epact Numbers': range(66272, 66303+1), 'Old Italic': range(66304, 66351+1), 'Gothic': range(66352, 66383+1), 'Old Permic': range(66384, 66431+1), 'Ugaritic': range(66432, 66463+1), 'Old Persian': range(66464, 66527+1), 'Deseret': range(66560, 66639+1), 'Shavian': range(66640, 66687+1), 'Osmanya': range(66688, 66735+1), 'Osage': range(66736, 66815+1), 'Elbasan': range(66816, 66863+1), 'Caucasian Albanian': range(66864, 66927+1), 'Linear A': range(67072, 67455+1), 'Cypriot Syllabary': range(67584, 67647+1), 'Imperial Aramaic': range(67648, 67679+1), 'Palmyrene': range(67680, 67711+1), 'Nabataean': range(67712, 67759+1), 'Hatran': range(67808, 67839+1), 'Phoenician': range(67840, 67871+1), 'Lydian': range(67872, 67903+1), 'Meroitic Hieroglyphs': range(67968, 67999+1), 'Meroitic Cursive': range(68000, 68095+1), 'Kharoshthi': range(68096, 68191+1), 'Old South Arabian': range(68192, 68223+1), 'Old North Arabian': range(68224, 68255+1), 'Manichaean': range(68288, 68351+1), 'Avestan': range(68352, 68415+1), 'Inscriptional Parthian': range(68416, 68447+1), 'Inscriptional Pahlavi': range(68448, 68479+1), 'Psalter Pahlavi': range(68480, 68527+1), 'Old Turkic': range(68608, 68687+1), 'Old Hungarian': range(68736, 68863+1), 'Rumi Numeral Symbols': range(69216, 69247+1), 'Brahmi': range(69632, 69759+1), 'Kaithi': range(69760, 69839+1), 'Sora Sompeng': range(69840, 69887+1), 'Chakma': range(69888, 69967+1), 'Mahajani': range(69968, 70015+1), 'Sharada': range(70016, 70111+1), 'Sinhala Archaic Numbers': range(70112, 70143+1), 'Khojki': range(70144, 70223+1), 'Multani': range(70272, 70319+1), 'Khudawadi': range(70320, 70399+1), 'Grantha': range(70400, 70527+1), 'Newa': range(70656, 70783+1), 'Tirhuta': range(70784, 70879+1), 'Siddham': range(71040, 71167+1), 'Modi': range(71168, 71263+1), 'Mongolian Supplement': range(71264, 71295+1), 'Takri': range(71296, 71375+1), 'Ahom': range(71424, 71487+1), 'Warang Citi': range(71840, 71935+1), 'Zanabazar Square': range(72192, 72271+1), 'Soyombo': range(72272, 72367+1), 'Pau Cin Hau': range(72384, 72447+1), 'Bhaiksuki': range(72704, 72815+1), 'Marchen': range(72816, 72895+1), 'Masaram Gondi': range(72960, 73055+1), 'Cuneiform': range(73728, 74751+1), 'Cuneiform Numbers and Punctuation': range(74752, 74879+1), 'Early Dynastic Cuneiform': range(74880, 75087+1), 'Egyptian Hieroglyphs': range(77824, 78895+1), 'Anatolian Hieroglyphs': range(82944, 83583+1), 'Bamum Supplement': range(92160, 92735+1), 'Mro': range(92736, 92783+1), 'Bassa Vah': range(92880, 92927+1), 'Pahawh Hmong': range(92928, 93071+1), 'Miao': range(93952, 94111+1), 'Ideographic Symbols and Punctuation': range(94176, 94207+1), 'Tangut': range(94208, 100351+1), 'Tangut Components': range(100352, 101119+1), 'Kana Supplement': range(110592, 110847+1), 'Kana Extended-A': range(110848, 110895+1), 'Nushu': range(110960, 111359+1), 'Duployan': range(113664, 113823+1), 'Shorthand Format Controls': range(113824, 113839+1), 'Byzantine Musical Symbols': range(118784, 119039+1), 'Musical Symbols': range(119040, 119295+1), 'Ancient Greek Musical Notation': range(119296, 119375+1), 'Tai Xuan Jing Symbols': range(119552, 119647+1), 'Counting Rod Numerals': range(119648, 119679+1), 'Mathematical Alphanumeric Symbols': range(119808, 120831+1), 'Sutton SignWriting': range(120832, 121519+1), 'Glagolitic Supplement': range(122880, 122927+1), 'Mende Kikakui': range(124928, 125151+1), 'Adlam': range(125184, 125279+1), 'Arabic Mathematical Alphabetic Symbols': range(126464, 126719+1), 'Mahjong Tiles': range(126976, 127023+1), 'Domino Tiles': range(127024, 127135+1), 'Playing Cards': range(127136, 127231+1), 'Enclosed Alphanumeric Supplement': range(127232, 127487+1), 'Enclosed Ideographic Supplement': range(127488, 127743+1), 'Miscellaneous Symbols and Pictographs': range(127744, 128511+1), 'Emoticons range(Emoji)': range(128512, 128591+1), 'Ornamental Dingbats': range(128592, 128639+1), 'Transport and Map Symbols': range(128640, 128767+1), 'Alchemical Symbols': range(128768, 128895+1), 'Geometric Shapes Extended': range(128896, 129023+1), 'Supplemental Arrows-C': range(129024, 129279+1), 'Supplemental Symbols and Pictographs': range(129280, 129535+1), 'CJK Unified Ideographs Extension B': range(131072, 173791+1), 'CJK Unified Ideographs Extension C': range(173824, 177983+1), 'CJK Unified Ideographs Extension D': range(177984, 178207+1), 'CJK Unified Ideographs Extension E': range(178208, 183983+1), 'CJK Unified Ideographs Extension F': range(183984, 191471+1), 'CJK Compatibility Ideographs Supplement': range(194560, 195103+1), 'Tags': range(917504, 917631+1), 'Variation Selectors Supplement': range(917760, 917999+1)}  # type: Dict[str, range]

UNICODE_SECONDARY_RANGE_KEYWORD = [  # type: List[str]
    'Supplement',
    'Extended',
    'Extensions',
    'Modifier',
    'Marks',
    'Punctuation',
    'Symbols',
    'Forms',
    'Operators',
    'Miscellaneous',
    'Drawing',
    'Block',
    'Shapes',
    'Supplemental',
    'Tags'
]

RE_POSSIBLE_ENCODING_INDICATION = compile(
    r'(?:(?:encoding)|(?:charset)|(?:coding))(?:[\:= ]{1,10})(?:[\"\']?)([a-zA-Z0-9\-_]+)(?:[\"\']?)',
    IGNORECASE
)

IANA_SUPPORTED = sorted(  # type: List[str]
    filter(
        lambda x: x.endswith("_codec") is False and x not in {"rot_13", "tactis", "mbcs"},
        list(set(aliases.values()))
    )
)

IANA_SUPPORTED_COUNT = len(IANA_SUPPORTED)  # type: int

# pre-computed code page that are similar using the function cp_similarity.
IANA_SUPPORTED_SIMILAR = {  # type: Dict[str, List[str]]
    "cp037": [
        "cp1026",
        "cp1140",
        "cp273",
        "cp500"
    ],
    "cp1026": [
        "cp037",
        "cp1140",
        "cp273",
        "cp500"
    ],
    "cp1125": [
        "cp866"
    ],
    "cp1140": [
        "cp037",
        "cp1026",
        "cp273",
        "cp500"
    ],
    "cp1250": [
        "iso8859_2"
    ],
    "cp1251": [
        "kz1048",
        "ptcp154"
    ],
    "cp1252": [
        "cp1258",
        "iso8859_15",
        "iso8859_9",
        "latin_1"
    ],
    "cp1253": [
        "iso8859_7"
    ],
    "cp1254": [
        "cp1258",
        "iso8859_15",
        "iso8859_9",
        "latin_1"
    ],
    "cp1257": [
        "iso8859_13"
    ],
    "cp1258": [
        "cp1252",
        "cp1254",
        "iso8859_9",
        "latin_1"
    ],
    "cp273": [
        "cp037",
        "cp1026",
        "cp1140",
        "cp500"
    ],
    "cp437": [
        "cp850",
        "cp858",
        "cp860",
        "cp861",
        "cp862",
        "cp863",
        "cp865"
    ],
    "cp500": [
        "cp037",
        "cp1026",
        "cp1140",
        "cp273"
    ],
    "cp850": [
        "cp437",
        "cp857",
        "cp858",
        "cp865"
    ],
    "cp857": [
        "cp850",
        "cp858",
        "cp865"
    ],
    "cp858": [
        "cp437",
        "cp850",
        "cp857",
        "cp865"
    ],
    "cp860": [
        "cp437",
        "cp861",
        "cp862",
        "cp863",
        "cp865"
    ],
    "cp861": [
        "cp437",
        "cp860",
        "cp862",
        "cp863",
        "cp865"
    ],
    "cp862": [
        "cp437",
        "cp860",
        "cp861",
        "cp863",
        "cp865"
    ],
    "cp863": [
        "cp437",
        "cp860",
        "cp861",
        "cp862",
        "cp865"
    ],
    "cp865": [
        "cp437",
        "cp850",
        "cp857",
        "cp858",
        "cp860",
        "cp861",
        "cp862",
        "cp863"
    ],
    "cp866": [
        "cp1125"
    ],
    "iso8859_10": [
        "iso8859_14",
        "iso8859_15",
        "iso8859_4",
        "iso8859_9",
        "latin_1"
    ],
    "iso8859_11": [
        "tis_620"
    ],
    "iso8859_13": [
        "cp1257"
    ],
    "iso8859_14": [
        "iso8859_10",
        "iso8859_15",
        "iso8859_16",
        "iso8859_3",
        "iso8859_9",
        "latin_1"
    ],
    "iso8859_15": [
        "cp1252",
        "cp1254",
        "iso8859_10",
        "iso8859_14",
        "iso8859_16",
        "iso8859_3",
        "iso8859_9",
        "latin_1"
    ],
    "iso8859_16": [
        "iso8859_14",
        "iso8859_15",
        "iso8859_2",
        "iso8859_3",
        "iso8859_9",
        "latin_1"
    ],
    "iso8859_2": [
        "cp1250",
        "iso8859_16",
        "iso8859_4"
    ],
    "iso8859_3": [
        "iso8859_14",
        "iso8859_15",
        "iso8859_16",
        "iso8859_9",
        "latin_1"
    ],
    "iso8859_4": [
        "iso8859_10",
        "iso8859_2",
        "iso8859_9",
        "latin_1"
    ],
    "iso8859_7": [
        "cp1253"
    ],
    "iso8859_9": [
        "cp1252",
        "cp1254",
        "cp1258",
        "iso8859_10",
        "iso8859_14",
        "iso8859_15",
        "iso8859_16",
        "iso8859_3",
        "iso8859_4",
        "latin_1"
    ],
    "kz1048": [
        "cp1251",
        "ptcp154"
    ],
    "latin_1": [
        "cp1252",
        "cp1254",
        "cp1258",
        "iso8859_10",
        "iso8859_14",
        "iso8859_15",
        "iso8859_16",
        "iso8859_3",
        "iso8859_4",
        "iso8859_9"
    ],
    "mac_iceland": [
        "mac_roman",
        "mac_turkish"
    ],
    "mac_roman": [
        "mac_iceland",
        "mac_turkish"
    ],
    "mac_turkish": [
        "mac_iceland",
        "mac_roman"
    ],
    "ptcp154": [
        "cp1251",
        "kz1048"
    ],
    "tis_620": [
        "iso8859_11"
    ]
}
