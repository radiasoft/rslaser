# -*- coding: utf-8 -*-
"""Routines to manage physical units
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""
import math

from rslaser_new.utils.string_tools import removeWhitespace
import scipy.constants as const

# How to use:
#
# convertUnitsNumber(5, "km", "m") returns 5000
# convertUnitsString("5 km", "m") returns "5000 m"
# convertUnitsNumberToString(5, "km", "m") returns "5000 m"
# convertUnitsStringToNumber("5 km", "m") returns 5000
# displayWithUnitsNumber(5000, "m") returns "5 km"
# displayWithUnitsString("5000 m") returns "5 km"
#
# The above functions should be used instead of __parseUnits
# or the __unitConversion dictionary
#
# __parseUnits("km") returns 1000


def __parseUnits(unit):
    # Attempt parsing of compound unit (e.g., 'm/s^2')
    convertValue = 1.0
    currentUnit = ""
    multiply = True

    for char in unit + "*":  # add extra '*' to process last unit
        if char in ["/", "*"]:
            if "^" in currentUnit:
                currentUnit, exponent = currentUnit.split("^")
                exponent = float(exponent)
            else:
                exponent = 1.0
            exponent = exponent if multiply else -exponent
            convertValue = convertValue * (__unitConversion[currentUnit] ** exponent)
            multiply = char == "*"
            currentUnit = ""
        else:
            currentUnit = currentUnit + char

    return convertValue


def separateNumberUnit(inputString):
    # How this works: after stripping all whitespace, the functions looks for
    # the largest continuous block of characters starting from the left that
    # can be converted to a float.  The rest (if any) are assumed to specify
    # the units.
    parse = removeWhitespace(inputString)

    for numLength in range(len(parse), -1, -1):
        try:
            number = float(parse[:numLength])
            unit = parse[numLength:]
            if unit.startswith("/"):
                unit = "1" + unit
                number = float(parse[: (numLength - 1)])
        except ValueError:
            continue
        else:
            break

    try:
        return number, unit
    except UnboundLocalError:
        return float(inputString), ""


def convertUnitsNumber(number, oldUnit, newUnit):
    oldUnit = removeWhitespace(oldUnit)
    newUnit = removeWhitespace(newUnit)

    if "" in [oldUnit, newUnit] and "%" not in [oldUnit, newUnit]:
        return number  # values without units don't get converted

    try:
        return number * __parseUnits(oldUnit) / __parseUnits(newUnit)
    except (ValueError, KeyError):
        raise ValueError('Cannot convert "' + oldUnit + '" to "' + newUnit + '".')


def convertUnitsString(inputString, newUnit):
    number, unit = separateNumberUnit(inputString)
    return convertUnitsNumberToString(number, unit, newUnit)


def convertUnitsNumberToString(number, oldUnit, newUnit):
    return (
        str(convertUnitsNumber(number, oldUnit, newUnit))
        + " "
        + removeWhitespace(newUnit)
    ).strip()


def convertUnitsStringToNumber(inputString, newUnit):
    value, unit = separateNumberUnit(inputString)
    return convertUnitsNumber(value, unit, newUnit)


# This function converts a value to units that result in the
# smallest number larger than one.
def displayWithUnitsNumber(value, currentUnit):
    if currentUnit is None:
        return str(value)
    if value == 0:
        return str(value) + " " + currentUnit
    if value < 0:
        return "-" + displayWithUnitsNumber(-value, currentUnit)

    # Separate compound units
    # 'ft/sec' -> 'ft' '/sec'
    restUnit = ""
    for symbol in ["/", "*"]:
        i = currentUnit.find(symbol)
        if i > -1:
            currentUnit, restUnit = currentUnit[:i], currentUnit[i:] + restUnit

    # Convert only first part of compound unit
    if "^" in currentUnit:
        baseUnit, exponent = currentUnit.split("^")
    else:
        baseUnit, exponent = currentUnit, None
    extra = "" if exponent is None else ("^" + exponent)
    try:
        group = unitTable[baseUnit]
        if "-" in extra:  # negative exponent
            group = reversed(group)
        for unit in [u + extra for u in group]:
            newValue = convertUnitsNumber(value, currentUnit, unit)
            if newValue >= 1:
                break
        return str(newValue) + " " + unit + restUnit
    except KeyError:
        return str(value) + " " + currentUnit


def displayWithUnitsString(inputString):
    value, unit = separateNumberUnit(inputString)
    return displayWithUnitsNumber(value, unit)


# Unit Conversions
__unitConversion = dict()
__unitConversion[""] = 1  # unitless unit
__unitConversion["1"] = 1  # for inverse units (1/s = Hz)
unitTable = dict()
prefixes = ["P", "T", "G", "M", "k", "", "m", "u", "n", "p", "f", "a"]
firstMultiplier = 1.0e15  # value of first unit prefix in prefixes


def addMetricUnit(unit, first=prefixes[0], last=prefixes[-1], addRow=True):
    multiplier = firstMultiplier
    row = []
    add = False
    for prefix in prefixes:
        if prefix == first:
            add = True
        if add:
            __unitConversion[prefix + unit] = multiplier
            row.append(prefix + unit)
        if prefix == last:
            break

        multiplier = multiplier / 1000

    if addRow:
        addToUnitTable(row)


def addToUnitConversion(unit, value, otherUnit):
    __unitConversion[unit] = value * __unitConversion[otherUnit]


def addToUnitTable(row):
    for unit in row:
        unitTable[unit] = row


def calculate_lambda0_from_phE(phE):
    return const.h * const.c / phE


def calculate_phE_from_lambda0(lambda0):
    return const.h * const.c / lambda0


# percent -> fraction
addToUnitConversion("%", 0.01, "")
addToUnitTable(["%"])

# length units -> meters
addMetricUnit("m", "k", "f", False)
addToUnitConversion("cm", 0.01, "m")
addToUnitConversion("micron", 1, "um")
addToUnitConversion("ang", 0.1, "nm")
addToUnitConversion("in", 2.54, "cm")
addToUnitConversion("mil", 0.001, "in")
addToUnitConversion("thou", 1, "mil")
addToUnitConversion("ft", 12, "in")
addToUnitConversion("yd", 3, "ft")
addToUnitConversion("mi", 5280, "ft")
addToUnitTable(["km", "m", "cm", "mm", "um", "nm", "pm", "fm"])
addToUnitTable(["mi", "yd", "ft", "in", "mil"])

# angle units -> rad
addMetricUnit("rad", "")
addToUnitConversion("deg", math.pi / 180, "rad")

# temporal frequency -> Hz
addMetricUnit("Hz")
addToUnitConversion("1/s", 1, "Hz")

# time -> seconds
addMetricUnit("s", "", "f", False)
addMetricUnit("sec", "", "f", False)
addToUnitConversion("min", 60, "sec")
addToUnitConversion("hr", 60, "min")
addToUnitTable(["hr", "min", "s", "ms", "us", "ns", "ps", "fs"])

# energy units -> eV
addMetricUnit("eV")

# charge units -> C
addMetricUnit("C")

# magnet units -> T
addMetricUnit("T")
addToUnitConversion("G", 0.0001, "T")
addToUnitConversion("mG", 0.001, "G")

# current -> A
addMetricUnit("A")

# energy -> J
addMetricUnit("J")

# power -> W
addMetricUnit("W")

# electrical resistance -> Ohm
addMetricUnit("Ohm")

# electrical potential -> V
addMetricUnit("V")

# mks mass -> g
addMetricUnit("g", "k")
