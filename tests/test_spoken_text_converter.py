"""
Unit tests for the SpokenTextConverter class.

This test suite validates the text-to-speech conversion functionality,
covering various text formats including percentages, times, currency,
years, decimal numbers, mixed text, special cases, and mathematical notation.
Each test function uses parameterized test cases to verify the accurate
conversion of text into its spoken form.
"""

# ruff: noqa: RUF001
from collections.abc import Generator

import pytest

from glados.utils.spoken_text_converter import SpokenTextConverter


@pytest.fixture
def converter() -> Generator[SpokenTextConverter, None, None]:
    """Provide a fresh SpokenTextConverter instance for each test."""
    yield SpokenTextConverter()


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("50%", "fifty percent"),
        ("3.5%", "three point five percent"),
        ("100.25%", "one hundred point two five percent"),
        ("0.5%", "zero point five percent"),
        ("1000%", "one thousand percent"),
        ("12.00%", "twelve percent"),
    ],
)
def test_convert_percentages(converter: SpokenTextConverter, input_text: str, expected: str) -> None:
    """
    Test the conversion of percentage strings to their spoken equivalents.

    This test function verifies that the `_convert_percentages` method of the SpokenTextConverter
    correctly transforms percentage representations into their spoken word format.

    Parameters:
        input_text (str): The percentage string to be converted (e.g., '50%', '25.5%')
        expected (str): The expected spoken representation of the percentage

    Raises:
        AssertionError: If the converted percentage does not match the expected spoken text
    """
    result = converter._convert_percentages(input_text)
    assert result.lower() == expected


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("3:00", "three o'clock"),
        ("5:05", "five oh five"),
        ("10:30", "ten thirty"),
        ("12:00", "twelve o'clock"),
        ("9:15", "nine fifteen"),
        ("11:45", "eleven forty-five"),
        ("3:00pm", "three in the afternoon"),
        ("7:30AM", "seven thirty in the morning"),
        ("8:05 pm", "eight oh five in the evening"),
    ],
)
def test_convert_times(converter: SpokenTextConverter, input_text: str, expected: str) -> None:
    """
    Test the conversion of time strings to their spoken equivalents.

    This test function verifies that the SpokenTextConverter correctly transforms
    various time representations into their spoken language format.

    Parameters:
        input_text (str): A time string to be converted (e.g., '3:45 PM', '14:30')
        expected (str): The expected spoken representation of the time

    Raises:
        AssertionError: If the converted time does not match the expected spoken text
    """
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("$1", "one dollar"),
        ("$5.25", "five dollars and twenty-five cents"),
        ("£1", "one pound"),
        ("£1.01", "one pound and one penny"),
        ("$1000", "one thousand dollars"),
        ("$0.50", "zero dollars and fifty cents"),
        ("$1.00", "one dollar"),
        ("£5.00", "five pounds"),
        ("$1000000", "one million dollars"),
        ("$1.99", "one dollar and ninety-nine cents"),
        ("£0.01", "zero pounds and one penny"),
        ("£99.99", "ninety-nine pounds and ninety-nine pence"),
    ],
)
def test_convert_currency(converter: SpokenTextConverter, input_text: str, expected: str) -> None:
    """
    Test the conversion of currency strings to their spoken text representation.

    This test function verifies that the SpokenTextConverter correctly transforms
    currency strings into their spoken language equivalents. It uses parametrized
    inputs to cover various currency formats and symbols.

    Parameters:
        input_text (str): A currency string to be converted (e.g., "$10.50", "€100")
        expected (str): The expected spoken text representation of the currency

    Raises:
        AssertionError: If the converted text does not match the expected spoken form
    """
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("1999", "nineteen ninety-nine"),
        ("2024", "twenty twenty-four"),
        ("2000s", "two thousands"),
        ("1805", "eighteen oh five"),
        ("1900", "nineteen hundred"),
        ("2000", "two thousand"),
        ("1066", "ten sixty-six"),
        ("1776", "seventeen seventy-six"),
        ("2100", "twenty-one hundred"),
        ("1950s", "nineteen fifties"),
    ],
)
def test_convert_years(converter: SpokenTextConverter, input_text: str, expected: str) -> None:
    """
    Test the conversion of year strings to their spoken text representation.

    This test function verifies that the SpokenTextConverter correctly transforms
    various year formats into their spoken language equivalents.

    Parameters:
        input_text (str): The input year string to be converted
        expected (str): The expected spoken text representation of the year

    Raises:
        AssertionError: If the converted text does not match the expected spoken form
    """
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("3.14", "three point one four"),
        ("0.5", "zero point five"),
        ("1.0", "one"),
        ("10.01", "ten point zero one"),
        ("100.001", "one hundred point zero zero one"),
        ("0.333", "zero point three three three"),
    ],
)
def test_convert_decimal_numbers(converter: SpokenTextConverter, input_text: str, expected: str) -> None:
    """
    Test the conversion of decimal numbers to their spoken text representation.

    This test function verifies that the SpokenTextConverter correctly transforms decimal number strings
    into their corresponding spoken language equivalents.

    Parameters:
        input_text (str): The decimal number string to be converted
        expected (str): The expected spoken text representation of the decimal number

    Raises:
        AssertionError: If the converted text does not match the expected spoken form
    """
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()


@pytest.mark.parametrize(
    "input_text, expected",
    [
        (
            "The meeting at 3:00pm on 1/1/2024 will cost $50.00.",
            "The meeting at three in the afternoon on january first, twenty twenty-four will cost fifty dollars.",
        ),
        (
            "In 1999, the company grew by 25% and made £1000000.",
            "In nineteen ninety-nine, the company grew by twenty-five percent and made one million pounds.",
        ),
        (
            "Temperature is 98.6° with 0.5% margin of error.",
            "Temperature is ninety-eight point six degrees with zero point five percent margin of error.",
        ),
        (
            "I am at a meeting from 9:00am to 5:00 costs $100.50.",
            "I am at a meeting from nine in the morning to five o'clock costs one hundred dollars and fifty cents.",
        ),
        ("8 is the square root of 64", "eight is the square root of sixty-four"),
        ("8^2 = 64", "eight to the power of two equals sixty-four"),
    ],
)
def test_convert_mixed_text(converter: SpokenTextConverter, input_text: str, expected: str) -> None:
    """
    Test the conversion of mixed text containing various elements into spoken language.

    This test function verifies that the SpokenTextConverter can correctly convert a text string
    containing multiple types of elements (such as times, dates, currency, numbers) into their
    spoken equivalents.

    Parameters:
        input_text (str): The input text containing mixed elements to be converted
        expected (str): The expected spoken representation of the input text

    Raises:
        AssertionError: If the converted text does not match the expected spoken representation
    """
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Dr. Smith", "Doctor Smith"),
        ("Mr. Jones", "Mister Jones"),
        ("Mrs. Brown", "Mrs Brown"),
        ("Ms. Davis", "Miss Davis"),
        ("100,000", "one hundred thousand"),
        ("1,000,000", "one million"),
        ("yeah", "ye'a"),
    ],
)
def test_convert_special_cases(converter: SpokenTextConverter, input_text: str, expected: str) -> None:
    """
    Test the conversion of special text cases to spoken language.

    This parameterized test function verifies the conversion of special text cases
    such as titles, formatted numbers, and other unique text patterns into their
    spoken equivalents using the SpokenTextConverter.

    Parameters:
        input_text (str): The input text containing special cases to be converted
        expected (str): The expected spoken language representation of the input text

    Raises:
        AssertionError: If the converted text does not match the expected spoken form
    """
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("8^2 = 64", "eight to the power of two equals sixty-four"),
        ("2^3 = 8", "two to the power of three equals eight"),
        ("√16 = 4", "square root of sixteen equals four"),
        ("∛27 = 3", "cube root of twenty-seven equals three"),
        ("5 + 3 = 8", "five plus three equals eight"),
        ("10 - 5 = 5", "ten minus five equals five"),
        ("4 × 2 = 8", "four times two equals eight"),
        ("8 ÷ 2 = 4", "eight divided by two equals four"),
        ("x^2 + y^2 = z^2", "x to the power of two plus y to the power of two equals z to the power of two"),
        ("15/3 = 5", "fifteen over three equals five"),
    ],
)
def test_convert_mathematical_notation(converter: SpokenTextConverter, input_text: str, expected: str) -> None:
    """
    Test the conversion of mathematical notation to spoken text.

    This parameterized test function verifies that mathematical expressions are correctly
    converted to their spoken equivalents using the SpokenTextConverter.

    Parameters:
        input_text (str): A mathematical expression or notation to be converted
        expected (str): The expected spoken representation of the mathematical expression

    Raises:
        AssertionError: If the converted text does not match the expected spoken text
        (case-insensitive comparison)
    """
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()
