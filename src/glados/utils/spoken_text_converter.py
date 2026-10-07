# ruff: noqa: RUF001, RUF002
"""Deterministic English normalization for the TTS phonemizer.

Numeric expressions are consumed whole, before word/acronym normalization. Decimal
and currency strings never pass through a float. Ambiguous slash dates use the existing
month/day ordering; unambiguous day/month dates are also recognized. Dates use
spoken month names and ordinals. Bare
four-digit values in the historical year range retain the existing year reading,
except when followed by a quantity unit. Possessive apostrophes are preserved.
"""

from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import math
import re
from typing import ClassVar

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = (
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
)
_MONTHS = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_NUMBER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+"
_UNSIGNED = rf"(?:{_NUMBER})"
_EUROPEAN = r"(?:[0-9]{1,3}(?:\.[0-9]{3})+,[0-9]{1,2}|[0-9]+,[0-9]{1,2})(?![0-9])"
_MONEY_NUMBER = rf"(?:{_EUROPEAN}|{_UNSIGNED})"
_SIGNED = rf"[+−-]?{_UNSIGNED}"
_MERIDIEM = r"[ap]\.?m\.?"
_UNITS = {
    "kg": ("kilogram", "kilograms"),
    "g": ("gram", "grams"),
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "ms": ("millisecond", "milliseconds"),
    "hz": ("hertz", "hertz"),
    "khz": ("kilohertz", "kilohertz"),
    "mhz": ("megahertz", "megahertz"),
    "ghz": ("gigahertz", "gigahertz"),
    "kb": ("kilobyte", "kilobytes"),
    "mb": ("megabyte", "megabytes"),
    "gb": ("gigabyte", "gigabytes"),
    "tb": ("terabyte", "terabytes"),
    "kib": ("kibibyte", "kibibytes"),
    "mib": ("mebibyte", "mebibytes"),
    "gib": ("gibibyte", "gibibytes"),
    "tib": ("tebibyte", "tebibytes"),
}
_UNIT_PATTERN = "|".join(sorted(_UNITS, key=len, reverse=True))
_UNIT_SUFFIX = rf"(?:°[ \t]*[CF]|℃|℉|°|{_UNIT_PATTERN})"
_TOKENS = re.compile(
    rf"(?<![\w.])(?=[0-9+−.$£€√∛-]|[a-zA-Z]\^)(?:"
    rf"(?P<iso>[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})(?![\w-])|"
    rf"(?P<date>[0-9]{{1,2}}/[0-9]{{1,2}}/(?:[0-9]{{4}}|[0-9]{{2}}))(?![\w/])|"
    rf"(?P<power>(?:{_SIGNED}|[a-zA-Z])\^{{1}}{_SIGNED})(?!\w)|"
    rf"(?P<root>[√∛]{_SIGNED})(?!\w)|"
    rf"(?P<fraction>{_SIGNED}/{_SIGNED})(?![\w/])|"
    rf"(?P<version>[0-9]+(?:\.[0-9]+){{2,}})(?!\w)|"
    rf"(?P<time>[0-9]{{1,2}}:[0-9]{{2}}(?::[0-9]{{2}})?(?:[ \t]*{_MERIDIEM})?)(?!\w)|"
    rf"(?P<spaced_time>[0-9]{{1,2}}[ \t]+[0-9]{{2}}[ \t]*{_MERIDIEM})(?!\w)|"
    rf"(?P<hour>[0-9]{{1,2}}[ \t]*{_MERIDIEM})(?!\w)|"
    rf"(?P<money>(?:[+−-]?[$£€]|[$£€][ \t]*[+−-]?)[ \t]*{_MONEY_NUMBER}"
    rf"(?:[ \t]*(?:hundred|thousand|million|billion|trillion|[kmb]))?)(?!\w)|"
    rf"(?P<ordinal>[0-9]+(?:st|nd|rd|th))(?!\w)|"
    rf"(?P<scientific>{_SIGNED}[eE][+-]?[0-9]+)(?!\w)|"
    rf"(?P<range>{_SIGNED}[ \t]*[-–—][ \t]*{_SIGNED}(?:[ \t]*{_UNIT_SUFFIX})?)(?!\w)|"
    rf"(?P<percent>{_SIGNED}[ \t]*%)(?!\w)|"
    rf"(?P<unit>{_SIGNED}[ \t]*{_UNIT_SUFFIX})(?!\w)|"
    rf"(?P<number>{_SIGNED})(?!\w))",
    re.IGNORECASE,
)
_RANGE = re.compile(rf"({_SIGNED})([ \t]*[-–—][ \t]*)({_SIGNED})(?:[ \t]*({_UNIT_SUFFIX}))?\Z", re.IGNORECASE)
_TIME = re.compile(
    rf"([0-9]{{1,2}})(?:(?::|[ \t]+)([0-9]{{2}})(?::([0-9]{{2}}))?)?[ \t]*({_MERIDIEM})?\Z", re.IGNORECASE
)
_UNIT = re.compile(rf"({_SIGNED})[ \t]*(.*)\Z")
_QUANTITY = re.compile(
    r"[ \t]+(?:tokens?|samples?|items?|people|persons?|files?|bytes?|dollars?|pounds?|euros?|"
    r"seconds?|minutes?|hours?|days?|meters?|metres?|kilometers?|kilometres?|grams?|kilograms?|"
    r"percent|degrees?|milliseconds?|megabytes?|gigabytes?|hertz|votes?|points?|tests?|rows?|requests?)\b",
    re.IGNORECASE,
)
_WORDS = re.compile(r"\b(?:[A-Z][ \t]+)+[A-Z]\b|\b[A-Za-z]+\b")
_ESCAPED_TEMPERATURE = re.compile(r"\\u(?:00b0|2103|2109)", re.IGNORECASE)
_TEMPERATURE_SYMBOL = re.compile(r"°[ \t]*[CF]\b|[℃℉]", re.IGNORECASE)
_SPACES = re.compile(r"[ \t]+")
_ELLIPSES = re.compile(r"\.{3,}|\. \. \.")
_TITLES = re.compile(r"\b(?:Dr|Mr|Mrs|Ms)\.(?=\s|$)", re.IGNORECASE)
_PUNCTUATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "ʼ": "'",
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "(": "«",
        ")": "»",
        "、": ", ",
        "。": ". ",
        "！": "! ",
        "，": ", ",
        "：": ": ",
        "；": "; ",
        "？": "? ",
        "…": "",
    }
)
_PROTECTED = re.compile(r"(?:https?://|www\.)[^\s<>]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
_MATH_OPERATOR = re.compile(r"(?<=[\w)])\s*([=+×÷])\s*(?=[\w(−-])")
_POWER = re.compile(rf"({_SIGNED}|[a-zA-Z])\^({_SIGNED})")
_ROOT = re.compile(rf"([√∛])({_SIGNED})")
_FRACTION = re.compile(rf"(?<![\w/])({_SIGNED})/({_SIGNED})(?![\w/])")
_PERCENT = re.compile(rf"(?<![\w.])({_SIGNED})[ \t]*%")
_VALID_NUMBER = re.compile(r"([+-]?)([0-9]*)(?:\.([0-9]+))?\Z")
_PAST_PARTICIPLE = re.compile(
    r"(?:already |just |never )?(?:been|done|gone|got|had|seen|known|taken|given|made|said|heard|"
    r"written|read|left|lost|found|bought|brought|forgotten|eaten|spoken|[a-z]+ed)\b",
    re.IGNORECASE,
)


def _integer_words(digits: str) -> str:
    digits = digits.lstrip("0") or "0"
    if len(digits) > 3 * len(_SCALES):
        # Do not cache arbitrarily long identifiers or exceed Python's int limit.
        return " ".join(_ONES[int(digit)] for digit in digits)
    return _small_integer_words(digits)


@lru_cache(maxsize=512)
def _small_integer_words(digits: str) -> str:
    value = int(digits)
    if not value:
        return "zero"
    chunks = []
    scale = 0
    while value:
        value, chunk = divmod(value, 1000)
        if chunk:
            words = []
            hundreds, rest = divmod(chunk, 100)
            if hundreds:
                words.append(f"{_ONES[hundreds]} hundred")
            if rest:
                if rest < 20:
                    words.append(_ONES[rest])
                else:
                    tens, ones = divmod(rest, 10)
                    words.append(_TENS[tens] + (f"-{_ONES[ones]}" if ones else ""))
            if scale:
                words.append(_SCALES[scale])
            chunks.append(" ".join(words))
        scale += 1
    return " ".join(reversed(chunks))


def _ordinal_words(digits: str) -> str:
    words = _integer_words(digits)
    return re.sub(
        r"[a-z]+$", lambda m: _ORDINALS.get(m[0], m[0][:-1] + "ieth" if m[0].endswith("y") else m[0] + "th"), words
    )


class SpokenTextConverter:
    """Normalize English text without loading models or making network requests."""

    CONTRACTIONS: ClassVar[dict[str, str]] = {
        "I'm": "I am",
        "I'll": "I will",
        "I've": "I have",
        "I'd": "I would",
        "won't": "won't",
        "can't": "can't",
        "shan't": "shan't",
        "ain't": "ain't",
        "let's": "let us",
        "'ll": " will",
        "'re": " are",
        "'ve": " have",
        "'m": " am",
        "'d": " would",
    }
    _CONTRACTIONS: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:(?:won't|can't|shan't|ain't)(?:'ve)?|(?:let's|y'all)|[a-z]+(?:n't(?:'ve)?|'ll|'re|'ve|'m|'d(?:'ve)?)"
        r"|(?:he|she|it|that|there|here|what|who|where|how)'s)\b",
        re.IGNORECASE,
    )
    convertible_pattern: ClassVar[re.Pattern[str]] = re.compile(r"[0-9$£€×÷^√∛=+−°]|\b(?:Dr|Mr|Mrs|Ms)\.|\.{3,}")

    def _number_to_words(self, num: float | str) -> str:
        """Keep numeric strings exact; accept finite numeric inputs for existing callers."""
        if isinstance(num, float):
            if not math.isfinite(num):
                raise ValueError(f"Invalid number format: {num}")
            text = format(Decimal(str(num)), "f")
        elif isinstance(num, str | int):
            text = str(num).strip().replace("−", "-").replace(",", "")
            if "e" in text.lower():
                try:
                    value = Decimal(text)
                    if not value.is_finite() or abs(value.adjusted()) > 1000:
                        raise ValueError(f"Invalid number format: {num}")
                    text = format(value, "f")
                except InvalidOperation as exc:
                    raise ValueError(f"Invalid number format: {num}") from exc
        else:
            raise ValueError(f"Invalid number format: {num}")
        match = _VALID_NUMBER.fullmatch(text)
        if not match or not (match[2] or match[3]):
            raise ValueError(f"Invalid number format: {num}")
        sign, integer, fraction = match.groups()
        result = _integer_words(integer or "0")
        fraction = (fraction or "").rstrip("0")
        if fraction:
            result += " point " + " ".join(_ONES[int(digit)] for digit in fraction)
        if sign == "-" and (fraction or (integer and integer.strip("0"))):
            result = "negative " + result
        elif sign == "+":
            result = "plus " + result
        return result

    @staticmethod
    def _year_words(digits: str) -> str:
        value = int(digits)
        if value < 1000:
            return _integer_words(digits)
        left, right = divmod(value, 100)
        if value == 2000:
            return "two thousand"
        if not right:
            return _integer_words(str(left)) + " hundred"
        return _integer_words(str(left)) + (" oh " if right < 10 else " ") + _integer_words(str(right))

    def _expand_contraction(self, match: re.Match[str]) -> str:
        word = match[0].lower()
        special = {
            "won't": "won't",
            "can't": "can't",
            "shan't": "shan't",
            "ain't": "ain't",
            "let's": "let us",
            "y'all": "you all",
        }
        if word.endswith("'ve") and word.count("'") > 1:
            base = word[:-3]
            if base.endswith("n't"):
                return base + " have"
            return ("I" if base[:-2] == "i" else base[:-2]) + " would have"
        if word.endswith("n't"):
            # Expanding an inverted question gives "can not you see?". Pronounce
            # the contraction itself; the phonemizer has entries for these words.
            return word
        if word in special:
            return special[word]
        stem, suffix = word.rsplit("'", 1)
        if suffix in {"s", "d"}:
            following = match.string[match.end() :].lstrip()
            past = bool(_PAST_PARTICIPLE.match(following))
            if suffix == "s" and re.match(
                r"(?:already |just |never )?(?:tired|bored|excited|interested|worried|confused|annoyed|"
                r"surprised|pleased|frustrated)\b",
                following,
                re.IGNORECASE,
            ):
                past = False
            if suffix == "d" and following.lower().startswith("better "):
                past = True
            if suffix == "s" and re.match(r"(?:already |just |never )?done\b", following, re.IGNORECASE):
                past = bool(
                    re.match(r"(?:already |just |never )?done (?:it|that|this|the|a)\b", following, re.IGNORECASE)
                )
            if suffix == "s" and following.lower().startswith("cost "):
                past = True
            # A participle can follow either "is" or "has" ("he's injured").
            # Keep ambiguous forms contracted instead of changing the sentence's meaning.
            if (
                suffix == "s"
                and past
                and not re.match(
                    r"(?:already |just |never )?(?:been|cost|done)\b",
                    following,
                    re.IGNORECASE,
                )
            ):
                return stem + "'s"
            if suffix == "d" and re.match(r"(?:already |just |never )?read\b", following, re.IGNORECASE):
                return ("I" if stem == "i" else stem) + "'d"
            expansion = ("has" if past else "is") if suffix == "s" else ("had" if past else "would")
        else:
            expansion = {"ll": "will", "re": "are", "ve": "have", "m": "am"}[suffix]
        return ("I" if stem == "i" else stem) + " " + expansion

    def _time_words(self, text: str) -> str:
        match = _TIME.fullmatch(text)
        if not match:
            return text
        hour, minute, second, meridiem = match.groups()
        h, m = int(hour), int(minute or "0")
        if not 0 <= m <= 59 or (second is not None and not 0 <= int(second) <= 59):
            return text
        if not (1 <= h <= 12 if meridiem else 0 <= h <= 23):
            return text
        result = _integer_words(hour)
        if m:
            result += (" oh " if m < 10 else " ") + _integer_words(str(m))
        elif not meridiem:
            result += " o'clock"
        if second is not None:
            result += " and " + _integer_words(second) + (" second" if int(second) == 1 else " seconds")
        if meridiem:
            if meridiem[0].lower() == "a":
                period = " at night" if h == 12 else " in the morning"
            else:
                period = " in the afternoon" if h == 12 or h < 6 else " in the evening"
            result += period
        return result

    def _money_words(self, text: str) -> str:
        symbol = next(char for char in text if char in "$£€")
        amount = text.replace(symbol, "").strip().replace("−", "-")
        scale_match = re.search(r"[ \t]*(hundred|thousand|million|billion|trillion|[kmb])$", amount, re.IGNORECASE)
        if scale_match:
            amount, scale = amount[: scale_match.start()].strip(), scale_match[1].lower()
            scale = {"k": "thousand", "m": "million", "b": "billion"}.get(scale, scale)
        else:
            scale = ""
        if re.fullmatch(rf"[+-]?{_EUROPEAN}", amount):
            amount = amount.replace(".", "").replace(",", ".")
        else:
            amount = amount.replace(",", "")
        sign = "negative " if amount.startswith("-") else "plus " if amount.startswith("+") else ""
        amount = amount.lstrip("+-")
        whole, _, fraction = amount.partition(".")
        whole = whole or "0"
        singular, plural, coin, coins = {
            "$": ("dollar", "dollars", "cent", "cents"),
            "£": ("pound", "pounds", "penny", "pence"),
            "€": ("euro", "euros", "cent", "cents"),
        }[symbol]
        if scale:
            return sign + self._number_to_words(amount) + " " + scale + " " + plural
        # More than two decimal places are precision, not hundreds of cents.
        if len(fraction) > 2 and fraction[2:].strip("0"):
            return sign + self._number_to_words(amount) + " " + plural
        cents = int((fraction[:2] + "00")[:2])
        result = sign + _integer_words(whole) + " " + (singular if whole.lstrip("0") == "1" else plural)
        if cents:
            result += " and " + _integer_words(str(cents)) + " " + (coin if cents == 1 else coins)
        return result

    def _spoken_token(self, match: re.Match[str]) -> str:
        text, kind = match[0], match.lastgroup
        if kind == "iso":
            try:
                value = date.fromisoformat(text)
            except ValueError:
                return text
            return f"{_MONTHS[value.month]} {_ordinal_words(str(value.day))}, {self._year_words(str(value.year))}"
        if kind == "date":
            month, day, year = text.split("/")
            if int(month) > 12:
                month, day = day, month
            year_value = int(year) if len(year) == 4 else 2000 + int(year)
            try:
                value = date(year_value, int(month), int(day))
            except ValueError:
                return text
            return f"{_MONTHS[value.month]} {_ordinal_words(str(value.day))}, {self._year_words(str(value.year))}"
        if kind in {"power", "root", "fraction"}:
            return self._convert_mathematical_notation(text)
        if kind == "version":
            return " point ".join(_integer_words(part) for part in text.split("."))
        if kind in {"time", "spaced_time", "hour"}:
            result = self._time_words(text)
            following = match.string[match.end() :].lstrip()
            if text.endswith(".") and (not following or following[0].isupper()):
                result += "."
            return result
        if kind == "money":
            return self._money_words(text)
        if kind == "ordinal":
            return _ordinal_words(text[:-2])
        if kind == "scientific":
            coefficient, exponent = re.split("[eE]", text)
            return self._number_to_words(coefficient) + " times ten to the power of " + self._number_to_words(exponent)
        if kind == "range":
            parts = _RANGE.fullmatch(text)
            assert parts is not None
            separator = (
                " minus "
                if parts[2].strip() == "-" and (" " in parts[2] or match.string[match.end() :].lstrip().startswith("="))
                else " to "
            )
            result = self._number_to_words(parts[1]) + separator + self._number_to_words(parts[3])
            if parts[4]:
                result += " " + self._unit_name(parts[3], parts[4])
            return result
        if kind == "percent":
            return self._number_to_words(text.rstrip("% \t")) + " percent"
        if kind == "unit":
            parts = _UNIT.fullmatch(text)
            assert parts is not None
            number, unit = parts.groups()
            words = self._number_to_words(number)
            return words + " " + self._unit_name(number, unit)
        if text.isdigit() and text.startswith("0") and len(text) > 1:
            return " ".join(_ONES[int(digit)] for digit in text)
        if (
            text.isdigit()
            and len(text) == 4
            and 1000 < int(text) < 3000
            and not _QUANTITY.match(match.string, match.end())
        ):
            return self._year_words(text)
        return _integer_words(text) if text.isdigit() else self._number_to_words(text)

    def _unit_name(self, number: str, unit: str) -> str:
        singular = self._number_to_words(number) in {"one", "negative one", "plus one"}
        if unit.startswith("°") or unit in {"℃", "℉"}:
            name = "degree" if singular else "degrees"
            if unit == "°":
                return name
            system = "celsius" if unit[-1].lower() == "c" or unit == "℃" else "fahrenheit"
            return name + " " + system
        return _UNITS[unit.lower()][0 if singular else 1]

    def _split_num(self, match: re.Match[str]) -> str:
        """Compatibility helper for existing time/year callers."""
        return self.text_to_spoken(match[0])

    def _flip_money(self, match: re.Match[str]) -> str:
        return self._money_words(match[0])

    def _point_num(self, match: re.Match[str]) -> str:
        return self._number_to_words(match[0])

    def _convert_percentages(self, text: str) -> str:
        return _PERCENT.sub(lambda m: self._number_to_words(m[1]) + " percent", text)

    def _contains_convertible_content(self, text: str) -> bool:
        return bool(self.convertible_pattern.search(text))

    def _convert_mathematical_notation(self, text: str) -> str:
        if "^" in text:
            text = _POWER.sub(
                lambda m: (m[1] if m[1].isalpha() else self._number_to_words(m[1]))
                + " to the power of "
                + self._number_to_words(m[2]),
                text,
            )
        if "√" in text or "∛" in text:
            text = _ROOT.sub(
                lambda m: ("square" if m[1] == "√" else "cube") + " root of " + self._number_to_words(m[2]),
                text,
            )
        if "/" in text:
            text = _FRACTION.sub(lambda m: self._number_to_words(m[1]) + " over " + self._number_to_words(m[2]), text)
        if any(symbol in text for symbol in "=+×÷"):
            text = _MATH_OPERATOR.sub(
                lambda m: " " + {"=": "equals", "+": "plus", "×": "times", "÷": "divided by"}[m[1]] + " ",
                text,
            )
        return text

    @staticmethod
    def _process_word(match: re.Match[str]) -> str:
        word = match[0]
        if word.lower() in {"kib", "mib", "gib", "tib"}:
            return _UNITS[word.lower()][1]
        if word == "IoT":
            return "I O T"
        if word.isupper() and len(word) > 1:
            return " ".join(word.split()) if " " in word or "\t" in word else " ".join(word)
        return word if word == "I" else word.lower()

    def _normalize_segment(self, text: str) -> str:
        if "'" in text:
            text = self._CONTRACTIONS.sub(self._expand_contraction, text)
        if self._contains_convertible_content(text):
            text = _TITLES.sub(
                lambda m: {"dr": "Doctor", "mr": "Mister", "mrs": "Mrs", "ms": "Miss"}[m[0][:-1].lower()], text
            )
            # Decades must be consumed before ordinary numeric tokens.
            text = re.sub(r"\b([12][0-9]{3})s\b", self._decade, text)
            text = _TOKENS.sub(self._spoken_token, text)
            text = self._convert_mathematical_notation(text)
        if "°" in text or "℃" in text or "℉" in text:
            # Numeric temperatures have already consumed the symbol. This also
            # handles a number written as words, as seen in generated replies.
            text = _TEMPERATURE_SYMBOL.sub(
                lambda m: " degrees " + ("celsius" if m[0][-1].lower() == "c" or m[0] == "℃" else "fahrenheit"),
                text,
            )
        text = _WORDS.sub(self._process_word, text)
        text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
        return re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)

    def _decade(self, match: re.Match[str]) -> str:
        value = int(match[1])
        if value == 2000:
            return "two thousands"
        left, right = divmod(value, 100)
        if not right:
            return _integer_words(str(left)) + " hundreds"
        last = _integer_words(str(right))
        return _integer_words(str(left)) + " " + (last[:-1] + "ies" if last.endswith("y") else last + "s")

    def text_to_spoken(self, text: str) -> str:
        """Convert a TTS segment; independent calls share only bounded integer caching."""
        if "\\u" in text:
            # Some model replies contain literal JSON escapes. Decode only known
            # temperature symbols, leaving paths and other backslashes alone.
            text = _ESCAPED_TEMPERATURE.sub(lambda m: chr(int(m[0][2:], 16)), text)
        if not text.isascii() or "(" in text or ")" in text:
            text = text.translate(_PUNCTUATION)
        text = _ELLIPSES.sub("", text)
        text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.splitlines() if line.strip())
        if "@" not in text and "://" not in text and "www." not in text.lower():
            return _SPACES.sub(" ", self._normalize_segment(text)).strip()
        # URLs and emails are opaque: punctuation inside them is not arithmetic.
        chunks, start = [], 0
        for match in _PROTECTED.finditer(text):
            chunks.append(self._normalize_segment(text[start : match.start()]))
            chunks.append(match[0])
            start = match.end()
        chunks.append(self._normalize_segment(text[start:]))
        return _SPACES.sub(" ", "".join(chunks)).strip()
