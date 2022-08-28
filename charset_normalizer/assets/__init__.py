# -*- coding: utf-8 -*-
from typing import Dict, List, Set

FREQUENCIES: Dict[str, List[str]] = {
    "English": [
        "e",
        "a",
        "t",
        "i",
        "o",
        "n",
        "s",
        "r",
        "h",
        "l",
        "d",
        "c",
        "u",
        "m",
        "f",
        "p",
        "g",
        "w",
        "y",
        "b",
        "v",
        "k",
        "x",
        "j",
        "z",
        "q",
    ],
    "German": [
        "e",
        "n",
        "i",
        "r",
        "s",
        "t",
        "a",
        "d",
        "h",
        "u",
        "l",
        "g",
        "o",
        "c",
        "m",
        "b",
        "f",
        "k",
        "w",
        "z",
        "p",
        "v",
        "ü",
        "ä",
        "ö",
        "j",
    ],
    "French": [
        "e",
        "a",
        "s",
        "n",
        "i",
        "t",
        "r",
        "l",
        "u",
        "o",
        "d",
        "c",
        "p",
        "m",
        "é",
        "v",
        "g",
        "f",
        "b",
        "h",
        "q",
        "à",
        "x",
        "è",
        "y",
        "j",
    ],
    "Dutch": [
        "e",
        "n",
        "a",
        "i",
        "r",
        "t",
        "o",
        "d",
        "s",
        "l",
        "g",
        "h",
        "v",
        "m",
        "u",
        "k",
        "c",
        "p",
        "b",
        "w",
        "j",
        "z",
        "f",
        "y",
        "x",
        "ë",
    ],
    "Italian": [
        "e",
        "i",
        "a",
        "o",
        "n",
        "l",
        "t",
        "r",
        "s",
        "c",
        "d",
        "u",
        "p",
        "m",
        "g",
        "v",
        "f",
        "b",
        "z",
        "h",
        "q",
        "è",
        "à",
        "k",
        "y",
        "ò",
    ],
    "Polish": [
        "a",
        "i",
        "o",
        "e",
        "n",
        "r",
        "z",
        "w",
        "s",
        "c",
        "t",
        "k",
        "y",
        "d",
        "p",
        "m",
        "u",
        "l",
        "j",
        "ł",
        "g",
        "b",
        "h",
        "ą",
        "ę",
        "ó",
    ],
    "Spanish": [
        "e",
        "a",
        "o",
        "n",
        "s",
        "r",
        "i",
        "l",
        "d",
        "t",
        "c",
        "u",
        "m",
        "p",
        "b",
        "g",
        "v",
        "f",
        "y",
        "ó",
        "h",
        "q",
        "í",
        "j",
        "z",
        "á",
    ],
    "Russian": [
        "о",
        "а",
        "е",
        "и",
        "н",
        "с",
        "т",
        "р",
        "в",
        "л",
        "к",
        "м",
        "д",
        "п",
        "у",
        "г",
        "я",
        "ы",
        "з",
        "б",
        "й",
        "ь",
        "ч",
        "х",
        "ж",
        "ц",
    ],
    "Japanese": [
        "の",
        "に",
        "る",
        "た",
        "は",
        "ー",
        "と",
        "し",
        "を",
        "で",
        "て",
        "が",
        "い",
        "ン",
        "れ",
        "な",
        "年",
        "ス",
        "っ",
        "ル",
        "か",
        "ら",
        "あ",
        "さ",
        "も",
        "り",
    ],
    "Portuguese": [
        "a",
        "e",
        "o",
        "s",
        "i",
        "r",
        "d",
        "n",
        "t",
        "m",
        "u",
        "c",
        "l",
        "p",
        "g",
        "v",
        "b",
        "f",
        "h",
        "ã",
        "q",
        "é",
        "ç",
        "á",
        "z",
        "í",
    ],
    "Swedish": [
        "e",
        "a",
        "n",
        "r",
        "t",
        "s",
        "i",
        "l",
        "d",
        "o",
        "m",
        "k",
        "g",
        "v",
        "h",
        "f",
        "u",
        "p",
        "ä",
        "c",
        "b",
        "ö",
        "å",
        "y",
        "j",
        "x",
    ],
    "Chinese": [
        "的",
        "一",
        "是",
        "不",
        "了",
        "在",
        "人",
        "有",
        "我",
        "他",
        "这",
        "个",
        "们",
        "中",
        "来",
        "上",
        "大",
        "为",
        "和",
        "国",
        "地",
        "到",
        "以",
        "说",
        "时",
        "要",
        "就",
        "出",
        "会",
    ],
    "Ukrainian": [
        "о",
        "а",
        "н",
        "і",
        "и",
        "р",
        "в",
        "т",
        "е",
        "с",
        "к",
        "л",
        "у",
        "д",
        "м",
        "п",
        "з",
        "я",
        "ь",
        "б",
        "г",
        "й",
        "ч",
        "х",
        "ц",
        "ї",
    ],
    "Norwegian": [
        "e",
        "r",
        "n",
        "t",
        "a",
        "s",
        "i",
        "o",
        "l",
        "d",
        "g",
        "k",
        "m",
        "v",
        "f",
        "p",
        "u",
        "b",
        "h",
        "å",
        "y",
        "j",
        "ø",
        "c",
        "æ",
        "w",
    ],
    "Finnish": [
        "a",
        "i",
        "n",
        "t",
        "e",
        "s",
        "l",
        "o",
        "u",
        "k",
        "ä",
        "m",
        "r",
        "v",
        "j",
        "h",
        "p",
        "y",
        "d",
        "ö",
        "g",
        "c",
        "b",
        "f",
        "w",
        "z",
    ],
    "Vietnamese": [
        "n",
        "h",
        "t",
        "i",
        "c",
        "g",
        "a",
        "o",
        "u",
        "m",
        "l",
        "r",
        "à",
        "đ",
        "s",
        "e",
        "v",
        "p",
        "b",
        "y",
        "ư",
        "d",
        "á",
        "k",
        "ộ",
        "ế",
    ],
    "Czech": [
        "o",
        "e",
        "a",
        "n",
        "t",
        "s",
        "i",
        "l",
        "v",
        "r",
        "k",
        "d",
        "u",
        "m",
        "p",
        "í",
        "c",
        "h",
        "z",
        "á",
        "y",
        "j",
        "b",
        "ě",
        "é",
        "ř",
    ],
    "Hungarian": [
        "e",
        "a",
        "t",
        "l",
        "s",
        "n",
        "k",
        "r",
        "i",
        "o",
        "z",
        "á",
        "é",
        "g",
        "m",
        "b",
        "y",
        "v",
        "d",
        "h",
        "u",
        "p",
        "j",
        "ö",
        "f",
        "c",
    ],
    "Korean": [
        "이",
        "다",
        "에",
        "의",
        "는",
        "로",
        "하",
        "을",
        "가",
        "고",
        "지",
        "서",
        "한",
        "은",
        "기",
        "으",
        "년",
        "대",
        "사",
        "시",
        "를",
        "리",
        "도",
        "인",
        "스",
        "일",
    ],
    "Indonesian": [
        "a",
        "n",
        "e",
        "i",
        "r",
        "t",
        "u",
        "s",
        "d",
        "k",
        "m",
        "l",
        "g",
        "p",
        "b",
        "o",
        "h",
        "y",
        "j",
        "c",
        "w",
        "f",
        "v",
        "z",
        "x",
        "q",
    ],
    "Turkish": [
        "a",
        "e",
        "i",
        "n",
        "r",
        "l",
        "ı",
        "k",
        "d",
        "t",
        "s",
        "m",
        "y",
        "u",
        "o",
        "b",
        "ü",
        "ş",
        "v",
        "g",
        "z",
        "h",
        "c",
        "p",
        "ç",
        "ğ",
    ],
    "Romanian": [
        "e",
        "i",
        "a",
        "r",
        "n",
        "t",
        "u",
        "l",
        "o",
        "c",
        "s",
        "d",
        "p",
        "m",
        "ă",
        "f",
        "v",
        "î",
        "g",
        "b",
        "ș",
        "ț",
        "z",
        "h",
        "â",
        "j",
    ],
    "Farsi": [
        "ا",
        "ی",
        "ر",
        "د",
        "ن",
        "ه",
        "و",
        "م",
        "ت",
        "ب",
        "س",
        "ل",
        "ک",
        "ش",
        "ز",
        "ف",
        "گ",
        "ع",
        "خ",
        "ق",
        "ج",
        "آ",
        "پ",
        "ح",
        "ط",
        "ص",
    ],
    "Arabic": [
        "ا",
        "ل",
        "ي",
        "م",
        "و",
        "ن",
        "ر",
        "ت",
        "ب",
        "ة",
        "ع",
        "د",
        "س",
        "ف",
        "ه",
        "ك",
        "ق",
        "أ",
        "ح",
        "ج",
        "ش",
        "ط",
        "ص",
        "ى",
        "خ",
        "إ",
    ],
    "Danish": [
        "e",
        "r",
        "n",
        "t",
        "a",
        "i",
        "s",
        "d",
        "l",
        "o",
        "g",
        "m",
        "k",
        "f",
        "v",
        "u",
        "b",
        "h",
        "p",
        "å",
        "y",
        "ø",
        "æ",
        "c",
        "j",
        "w",
    ],
    "Serbian": [
        "а",
        "и",
        "о",
        "е",
        "н",
        "р",
        "с",
        "у",
        "т",
        "к",
        "ј",
        "в",
        "д",
        "м",
        "п",
        "л",
        "г",
        "з",
        "б",
        "a",
        "i",
        "e",
        "o",
        "n",
        "ц",
        "ш",
    ],
    "Lithuanian": [
        "i",
        "a",
        "s",
        "o",
        "r",
        "e",
        "t",
        "n",
        "u",
        "k",
        "m",
        "l",
        "p",
        "v",
        "d",
        "j",
        "g",
        "ė",
        "b",
        "y",
        "ų",
        "š",
        "ž",
        "c",
        "ą",
        "į",
    ],
    "Slovene": [
        "e",
        "a",
        "i",
        "o",
        "n",
        "r",
        "s",
        "l",
        "t",
        "j",
        "v",
        "k",
        "d",
        "p",
        "m",
        "u",
        "z",
        "b",
        "g",
        "h",
        "č",
        "c",
        "š",
        "ž",
        "f",
        "y",
    ],
    "Slovak": [
        "o",
        "a",
        "e",
        "n",
        "i",
        "r",
        "v",
        "t",
        "s",
        "l",
        "k",
        "d",
        "m",
        "p",
        "u",
        "c",
        "h",
        "j",
        "b",
        "z",
        "á",
        "y",
        "ý",
        "í",
        "č",
        "é",
    ],
    "Hebrew": [
        "י",
        "ו",
        "ה",
        "ל",
        "ר",
        "ב",
        "ת",
        "מ",
        "א",
        "ש",
        "נ",
        "ע",
        "ם",
        "ד",
        "ק",
        "ח",
        "פ",
        "ס",
        "כ",
        "ג",
        "ט",
        "צ",
        "ן",
        "ז",
        "ך",
    ],
    "Bulgarian": [
        "а",
        "и",
        "о",
        "е",
        "н",
        "т",
        "р",
        "с",
        "в",
        "л",
        "к",
        "д",
        "п",
        "м",
        "з",
        "г",
        "я",
        "ъ",
        "у",
        "б",
        "ч",
        "ц",
        "й",
        "ж",
        "щ",
        "х",
    ],
    "Croatian": [
        "a",
        "i",
        "o",
        "e",
        "n",
        "r",
        "j",
        "s",
        "t",
        "u",
        "k",
        "l",
        "v",
        "d",
        "m",
        "p",
        "g",
        "z",
        "b",
        "c",
        "č",
        "h",
        "š",
        "ž",
        "ć",
        "f",
    ],
    "Hindi": [
        "क",
        "र",
        "स",
        "न",
        "त",
        "म",
        "ह",
        "प",
        "य",
        "ल",
        "व",
        "ज",
        "द",
        "ग",
        "ब",
        "श",
        "ट",
        "अ",
        "ए",
        "थ",
        "भ",
        "ड",
        "च",
        "ध",
        "ष",
        "इ",
    ],
    "Estonian": [
        "a",
        "i",
        "e",
        "s",
        "t",
        "l",
        "u",
        "n",
        "o",
        "k",
        "r",
        "d",
        "m",
        "v",
        "g",
        "p",
        "j",
        "h",
        "ä",
        "b",
        "õ",
        "ü",
        "f",
        "c",
        "ö",
        "y",
    ],
    "Simple English": [
        "e",
        "a",
        "t",
        "i",
        "o",
        "n",
        "s",
        "r",
        "h",
        "l",
        "d",
        "c",
        "m",
        "u",
        "f",
        "p",
        "g",
        "w",
        "b",
        "y",
        "v",
        "k",
        "j",
        "x",
        "z",
        "q",
    ],
    "Thai": [
        "า",
        "น",
        "ร",
        "อ",
        "ก",
        "เ",
        "ง",
        "ม",
        "ย",
        "ล",
        "ว",
        "ด",
        "ท",
        "ส",
        "ต",
        "ะ",
        "ป",
        "บ",
        "ค",
        "ห",
        "แ",
        "จ",
        "พ",
        "ช",
        "ข",
        "ใ",
    ],
    "Greek": [
        "α",
        "τ",
        "ο",
        "ι",
        "ε",
        "ν",
        "ρ",
        "σ",
        "κ",
        "η",
        "π",
        "ς",
        "υ",
        "μ",
        "λ",
        "ί",
        "ό",
        "ά",
        "γ",
        "έ",
        "δ",
        "ή",
        "ω",
        "χ",
        "θ",
        "ύ",
    ],
    "Tamil": [
        "க",
        "த",
        "ப",
        "ட",
        "ர",
        "ம",
        "ல",
        "ன",
        "வ",
        "ற",
        "ய",
        "ள",
        "ச",
        "ந",
        "இ",
        "ண",
        "அ",
        "ஆ",
        "ழ",
        "ங",
        "எ",
        "உ",
        "ஒ",
        "ஸ",
    ],
    "Classical Chinese": [
        "之",
        "年",
        "為",
        "也",
        "以",
        "一",
        "人",
        "其",
        "者",
        "國",
        "有",
        "二",
        "十",
        "於",
        "曰",
        "三",
        "不",
        "大",
        "而",
        "子",
        "中",
        "五",
        "四",
    ],
    "Kazakh": [
        "а",
        "ы",
        "е",
        "н",
        "т",
        "р",
        "л",
        "і",
        "д",
        "с",
        "м",
        "қ",
        "к",
        "о",
        "б",
        "и",
        "у",
        "ғ",
        "ж",
        "ң",
        "з",
        "ш",
        "й",
        "п",
        "г",
        "ө",
    ],
}


"""
FREQUENCIES_INDEX["Simple English"] =
{
    "e": 0,
    "a": 1,
    "t": 2,
    and so on...
}
"""
FREQUENCIES_INDEX: Dict[str, Dict[str, int]] = {
    key: {char: i for i, char in enumerate(value)} for key, value in FREQUENCIES.items()
}


FREQUENCIES_SETS: Dict[str, Set[str]] = {
    key: set(value) for key, value in FREQUENCIES.items()
}
