# -*- coding: utf-8 -*-
u"""Convenient string manipulation routines
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""
import string

# Divides a string into lines of maximum width 'lineWidth'. 'endLine' specifies
# a string to be appended to any wrapped lines if a continuation character is
# needed. The variable 'indenting' specifies a number of spaces to indent the
# wrapped lines.
def wordwrap(line, lineWidth, endLine = '', indenting = 0):
    if indenting > lineWidth/2:
        indenting = lineWidth/2
    line = line.replace('\n',' ') # get rid of any existing newlines
    lineBegin = 0 # location of last line break
    newLine = endLine + '\n'
    indent = ' '*indenting

    # word wrap: maximum line length is lineWidth
    while lineBegin + lineWidth < len(line):
        # location of editing cursor
        lineEdit = lineBegin + lineWidth - len(endLine)
        while (line[lineEdit] not in string.whitespace or insideQuote(line, lineEdit)) and lineEdit > lineBegin:
            lineEdit -= 1 # backup up until whitespace is found
        if lineEdit == lineBegin:
            # whitespace not found, skip ahead to next whitespace or end of line
            while (line[lineEdit] not in string.whitespace or insideQuote(line, lineEdit)) and lineEdit < len(line):
                lineEdit += 1
            if lineEdit == len(line):
                return line
        line = line[:lineEdit] + newLine + indent + line[lineEdit:].strip()
        lineBegin = lineEdit + len(newLine)

    return line

def insideQuote(line, position):
    quoted = False
    for index in range(position + 1):
        if line[index] == '"' and not characterEscaped(line, index):
            quoted = not quoted
    return quoted


def stripComments(line, commentCharacter):
    for i in range(len(line)):
        if line[i] == commentCharacter and not insideQuote(line, i) and not characterEscaped(line, i):
            return line[:i].strip()
    return line.strip()

def characterEscaped(line, position):
    return position > 0 \
            and line[position - 1] == '\\' \
            and not characterEscaped(line, position - 1)

def removeWhitespace(line):
    return ''.join(line.split())

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
