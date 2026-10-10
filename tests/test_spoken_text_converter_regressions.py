# ruff: noqa: RUF001
"""Regression cases from realistic TTS input, including interacting token types."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from glados.utils.spoken_text_converter import SpokenTextConverter


@pytest.fixture
def converter() -> SpokenTextConverter:
    return SpokenTextConverter()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I’m here. You’re back.", "I am here. you are back."),
        ("Iʼve finished; we‘ll leave.", "I have finished; we will leave."),
        ("Can't, CAN'T, won't, WON'T, shan't.", "can't, can't, won't, won't, shan't."),
        ("I'm ready. I'M ready. i'm ready.", "I am ready. I am ready. I am ready."),
        (
            "Don't. Doesn't. Didn't. Isn't. Aren't. Wasn't. Weren't.",
            "don't. doesn't. didn't. isn't. aren't. wasn't. weren't.",
        ),
        ("We've tried. They've tried. YOU'RE welcome.", "we have tried. they have tried. you are welcome."),
        ("Let's try. Y'all can help.", "let us try. you all can help."),
        ("It’s raining. That's fine. There's time.", "it is raining. that is fine. there is time."),
        ("He's been waiting. It's already done.", "he has been waiting. it is already done."),
        ("He's tired. She's bored. It's just confused.", "he is tired. she is bored. it is just confused."),
        (
            "I'd like tea. I'd already finished. I'd better go. I'd rather stay.",
            "I would like tea. I had already finished. I had better go. I would rather stay.",
        ),
        (
            "shouldn't've, couldn't've, wouldn't've, won't've",
            "shouldn't have, couldn't have, wouldn't have, won't have",
        ),
        ("I'd've gone. He'd've known.", "I would have gone. he would have known."),
        ("John's cup and the dog's bowl. James’ coat.", "john's cup and the dog's bowl. james' coat."),
        ("O'Neill's name and rock 'n' roll.", "o'neill's name and rock 'n' roll."),
        ("He's injured. It's completed. She's taken the test.", "he's injured. it's completed. she's taken the test."),
        ("I'd read it. He'd read it.", "I'd read it. he'd read it."),
        (
            "I'm using 4096 tokens; it's cost $1,234.56.",
            "I am using four thousand ninety-six tokens; it has cost one thousand two hundred thirty-four "
            "dollars and fifty-six cents.",
        ),
    ],
)
def test_contractions_and_possessives(converter: SpokenTextConverter, text: str, expected: str) -> None:
    assert converter.text_to_spoken(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Can't you see?", "can't you see?"),
        ("Can’t you see?", "can't you see?"),
        ("Don't you see?", "don't you see?"),
        ("Won't you come in?", "won't you come in?"),
        ("Isn't it obvious?", "isn't it obvious?"),
        ("Why haven't you finished?", "why haven't you finished?"),
        ("You can't do that.", "you can't do that."),
        ("I won't do that.", "I won't do that."),
        ("Don't touch that!", "don't touch that!"),
        ("Shouldn't you have checked?", "shouldn't you have checked?"),
        ("I wouldn't've said that.", "I wouldn't have said that."),
    ],
)
def test_negative_contractions_keep_natural_word_order(
    converter: SpokenTextConverter,
    text: str,
    expected: str,
) -> None:
    assert converter.text_to_spoken(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("It is 5 15 PM.", "it is five fifteen in the afternoon."),
        (
            "8 05 AM, 12 00 pm, 7 30 p.m.",
            "eight oh five in the morning, twelve in the afternoon, seven thirty in the evening.",
        ),
        ("I bought 5 15-inch displays.", "I bought five fifteen-inch displays."),
        ("There are 5 15 25 items.", "there are five fifteen twenty-five items."),
        (r"67\u00b0C", "sixty-seven degrees celsius"),
        (r"-1\u00B0F", "negative one degree fahrenheit"),
        (r"sixty-seven\u00b0C", "sixty-seven degrees celsius"),
        (r"20\u2103 and 70\u2109", "twenty degrees celsius and seventy degrees fahrenheit"),
        (r"Path C:\users\me", r"path c:\users\me"),
        ("1 MiB, 2 GiB, 0.5 TiB", "one mebibyte, two gibibytes, zero point five tebibytes"),
        ("1024 KiB", "one thousand twenty-four kibibytes"),
        ("9035 MiB", "nine thousand thirty-five mebibytes"),
        ("1132 MiB", "one thousand one hundred thirty-two mebibytes"),
        ("fifty-five point zero four gib", "fifty-five point zero four gibibytes"),
        ("CPU, GPU and RAM", "C P U, G P U and R A M"),
        ("C P U, G P U and R A M", "C P U, G P U and R A M"),
        ("A CPU and a GPU", "a C P U and a G P U"),
        ("IoT projects", "I O T projects"),
    ],
)
def test_session_edge_case_variants(converter: SpokenTextConverter, text: str, expected: str) -> None:
    assert converter.text_to_spoken(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1234.56", "one thousand two hundred thirty-four point five six"),
        ("2024.001", "two thousand twenty-four point zero zero one"),
        ("1,234.56", "one thousand two hundred thirty-four point five six"),
        ("1,000.05", "one thousand point zero five"),
        ("-17", "negative seventeen"),
        ("−3.5", "negative three point five"),
        ("-.5", "negative zero point five"),
        ("+2.5", "plus two point five"),
        (".000000000001", "zero point zero zero zero zero zero zero zero zero zero zero zero one"),
        ("9.876543210123", "nine point eight seven six five four three two one zero one two three"),
        ("1.23000", "one point two three"),
        ("-0.0", "zero"),
        ("1000000000000", "one trillion"),
        ("1,000,000,000,000", "one trillion"),
        (
            "9007199254740993",
            "nine quadrillion seven trillion one hundred ninety-nine billion two hundred fifty-four million "
            "seven hundred forty thousand nine hundred ninety-three",
        ),
        (
            "9007199254740993.01",
            "nine quadrillion seven trillion one hundred ninety-nine billion two hundred fifty-four million "
            "seven hundred forty thousand nine hundred ninety-three point zero one",
        ),
        ("Room 007; code 01234.", "room zero zero seven; code zero one two three four."),
        ("Values: 1,2,3.", "values: one,two,three."),
        ("12,34", "twelve,thirty-four"),
        (
            "4096 tokens; 2010 samples; 1234 people.",
            "four thousand ninety-six tokens; two thousand ten samples; one thousand two hundred thirty-four people.",
        ),
        ("In 2026 we return to 1905.", "in twenty twenty-six we return to nineteen oh five."),
        ("2000s and 2020s", "two thousands and twenty twenties"),
        (
            "21st, 22nd, 23rd, 24th, 100th, 101st",
            "twenty-first, twenty-second, twenty-third, twenty-fourth, one hundredth, one hundred first",
        ),
        ("11th 12th 13th 20th 30th 40th 80th", "eleventh twelfth thirteenth twentieth thirtieth fortieth eightieth"),
        ("Version 1.2.3", "version one point two point three"),
        ("1e-3", "one times ten to the power of negative three"),
        ("-2.5E+10", "negative two point five times ten to the power of plus ten"),
        ("v2.0 and abc123", "v2.0 and abc123"),
    ],
)
def test_numeric_expressions(converter: SpokenTextConverter, text: str, expected: str) -> None:
    assert converter.text_to_spoken(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$1,234.56", "one thousand two hundred thirty-four dollars and fifty-six cents"),
        ("£1,000.01", "one thousand pounds and one penny"),
        ("€1.05", "one euro and five cents"),
        ("€2.01", "two euros and one cent"),
        ("€1.234,56", "one thousand two hundred thirty-four euros and fifty-six cents"),
        ("€1,23", "one euro and twenty-three cents"),
        ("€1,234", "one thousand two hundred thirty-four euros"),
        ("$1234.56", "one thousand two hundred thirty-four dollars and fifty-six cents"),
        ("€-1.234,56", "negative one thousand two hundred thirty-four euros and fifty-six cents"),
        ("$.50", "zero dollars and fifty cents"),
        ("£.01", "zero pounds and one penny"),
        ("-$5.25", "negative five dollars and twenty-five cents"),
        ("$-5.25", "negative five dollars and twenty-five cents"),
        ("£ -1.50", "negative one pound and fifty pence"),
        ("$0.001", "zero point zero zero one dollars"),
        ("$1.000", "one dollar"),
        ("$1.999", "one point nine nine nine dollars"),
        ("$1.5 million", "one point five million dollars"),
        ("$2.5B", "two point five billion dollars"),
        ("€5k", "five thousand euros"),
        ("€1,23 million", "one point two three million euros"),
        ("€1.234,56M", "one thousand two hundred thirty-four point five six million euros"),
        ("-3.5%", "negative three point five percent"),
        ("+.5%", "plus zero point five percent"),
        ("1,234.5 %", "one thousand two hundred thirty-four point five percent"),
        ("100.00000000001%", "one hundred point zero zero zero zero zero zero zero zero zero zero one percent"),
        (
            "$1,234.56 and 25% in 2026",
            "one thousand two hundred thirty-four dollars and fifty-six cents and twenty-five percent in "
            "twenty twenty-six",
        ),
    ],
)
def test_currency_and_percentages(converter: SpokenTextConverter, text: str, expected: str) -> None:
    assert converter.text_to_spoken(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3:00 PM", "three in the afternoon"),
        ("3pm", "three in the afternoon"),
        ("9 AM", "nine in the morning"),
        ("12:05 a.m.", "twelve oh five at night."),
        ("At 3 p.m. we meet.", "at three in the afternoon we meet."),
        ("At 3 p.m. Then we leave.", "at three in the afternoon. then we leave."),
        ("12:00:30", "twelve o'clock and thirty seconds"),
        ("12:05:01 PM", "twelve oh five and one second in the afternoon"),
        ("00:00", "zero o'clock"),
        ("23:59", "twenty-three fifty-nine"),
        # Invalid dates and times are read as their numbers, never as an invented
        # date, and never left as digits: the phonemizer drops digits entirely.
        ("25:00", "twenty-five hundred"),
        ("12:99", "twelve ninety-nine"),
        ("13pm", "thirteen pm"),
        ("2026-10-07", "october seventh, twenty twenty-six"),
        ("2024-02-29", "february twenty-ninth, twenty twenty-four"),
        ("0001-01-01", "january first, one"),
        ("2026-02-30", "twenty twenty-six two thirty"),
        ("1/1/2005", "january first, twenty oh five"),
        ("31/12/2026", "december thirty-first, twenty twenty-six"),
        ("1/1/23", "january first, twenty twenty-three"),
        ("12/25/2000", "december twenty-fifth, two thousand"),
        ("4/5/2026", "april fifth, twenty twenty-six"),
        ("11/31/2024", "eleven thirty-one twenty twenty-four"),
        (
            "We'll meet at 3:00 PM on 2026-10-07, costing €1,234.56.",
            "we will meet at three in the afternoon on october seventh, twenty twenty-six, costing one thousand two "
            "hundred thirty-four euros and fifty-six cents.",
        ),
    ],
)
def test_times_and_dates(converter: SpokenTextConverter, text: str, expected: str) -> None:
    assert converter.text_to_spoken(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("-5°C to 10°C", "negative five degrees celsius to ten degrees celsius"),
        (
            "1°C, -1°F, 20℃ and 70℉",
            "one degree celsius, negative one degree fahrenheit, twenty degrees celsius and seventy degrees fahrenheit",
        ),
        ("98.6°", "ninety-eight point six degrees"),
        ("3-5 minutes", "three to five minutes"),
        ("1–3", "one to three"),
        ("-3--1", "negative three to negative one"),
        ("3-5kg", "three to five kilograms"),
        ("5–10°C", "five to ten degrees celsius"),
        ("32GB and 1kg", "thirty-two gigabytes and one kilogram"),
        ("1.0kg and 1,000Hz", "one kilogram and one thousand hertz"),
        ("5+3=8", "five plus three equals eight"),
        ("10-5=5", "ten minus five equals five"),
        ("10 - 5", "ten minus five"),
        ("Range 3-5, x=1.", "range three to five, x equals one."),
        ("3.5/7 = 0.5", "three point five over seven equals zero point five"),
        ("x^-2 and √0.25", "x to the power of negative two and square root of zero point two five"),
        ("A well-known test - please use 5 samples.", "a well-known test - please use five samples."),
        (
            "Read https://example.com/2026/10?q=1+2 and email test+1@example.com.",
            "read https://example.com/2026/10?q=1+2 and email test+1@example.com.",
        ),
        ("First line.\n\n  Second\tline has 2 items.\n", "first line.\nsecond line has two items."),
        ("“I’m ready…” (yes)", '"I am ready" «yes»'),
        ("", ""),
        ("  \n \t", ""),
    ],
)
def test_units_ranges_and_text_boundaries(converter: SpokenTextConverter, text: str, expected: str) -> None:
    assert converter.text_to_spoken(text) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (42, "forty-two"),
        (-17, "negative seventeen"),
        (
            "9007199254740993",
            "nine quadrillion seven trillion one hundred ninety-nine billion two hundred fifty-four million "
            "seven hundred forty thousand nine hundred ninety-three",
        ),
        (
            "0.1234567890123456789",
            "zero point one two three four five six seven eight nine zero one two three four five six seven eight nine",
        ),
        (0.00000000001, "zero point zero zero zero zero zero zero zero zero zero zero one"),
        ("1e3", "one thousand"),
        ("-0", "zero"),
    ],
)
def test_number_helper_keeps_precision(converter: SpokenTextConverter, number: float | str, expected: str) -> None:
    assert converter._number_to_words(number) == expected


@pytest.mark.parametrize("number", ["not a number", "nan", "inf", "", None, float("inf"), float("nan")])
def test_invalid_number_helpers_raise_value_error(converter: SpokenTextConverter, number: float | str | None) -> None:
    with pytest.raises(ValueError, match="Invalid number format"):
        converter._number_to_words(number)


def test_very_large_number_does_not_overflow(converter: SpokenTextConverter) -> None:
    assert converter.text_to_spoken("9" * 4500).split() == ["nine"] * 4500


def test_shared_converter_keeps_requests_independent(converter: SpokenTextConverter) -> None:
    cases = ["$1,234.56", "I'd already finished.", "-5°C", "5–10°C", "3:00 PM", "1234.56"] * 20
    expected = [converter.text_to_spoken(text) for text in cases]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(converter.text_to_spoken, cases)) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # "-ed" verbs in the present tense keep "would"; real participles take "had".
        ("I'd need a minute.", "I would need a minute."),
        ("We'd succeed.", "we would succeed."),
        ("She'd feed the cat.", "she would feed the cat."),
        ("I'd finished it. I'd agreed.", "I had finished it. I had agreed."),
        # Units are case-sensitive: network generations and acronyms are not grams or milliseconds.
        ("5G networks", "five g networks"),
        ("4G LTE", "four g L T E"),
        ("Top 10 MS products", "top ten M S products"),
        ("use 2FA", "use two F A"),
        ("It weighs 5 g. 5GB free.", "it weighs five grams. five gigabytes free."),
        ("Wait 200 ms.", "wait two hundred milliseconds."),
        ("3.5 GHz, 16 GB", "three point five gigahertz, sixteen gigabytes"),
        ("Ms. Smith and Dr. Who", "miss smith and doctor who"),
        # Year spans are read as years; phone numbers digit by digit; other spans as ranges.
        ("2024-2025 season", "twenty twenty-four to twenty twenty-five season"),
        ("year 1984-85", "year nineteen eighty-four to eighty-five"),
        ("Call 555-1234", "call five five five, one two three four"),
        ("1-800-555-1234", "one, eight zero zero, five five five, one two three four"),
        ("pages 10-20", "pages ten to twenty"),
    ],
)
def test_review_regressions(converter: SpokenTextConverter, text: str, expected: str) -> None:
    assert converter.text_to_spoken(text) == expected
