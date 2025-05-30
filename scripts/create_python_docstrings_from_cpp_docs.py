#!/usr/bin/env python3

import re


def camel_case_to_snake_case(text):
    """Converts CamelCase to snake_case"""
    return re.sub(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))", r"_\1", text).lower()


FILES_TO_PARSE = [
    {"filepath": "docs/api/cpp/doom_game.md", "namespace": "DoomGamePython"},
    {"filepath": "docs/api/cpp/utils.md"},
]
OUTPUT_FILE = "src/lib_python/ViZDoomMethodsDocstrings.h"
RAW_STRING_ESCAPE_SEQ = "DOCSTRING"
FUNCTION_HEADER_REGEX = r"^##+ *`([a-zA-Z]+)` *$"
TO_REPLACE = [
    ("true", "`True`"),  # Cpp -> Python bool
    ("false", "`False`"),  # Cpp -> Python bool
    ("`nullptr/null/None`", "`None`"),  # Cpp -> Python None
    (
        r"\[(`[a-z][a-zA-Z]+`)?\]\(.*?\)",
        r":meth:\1",
    ),  # MD methods links -> :meth:`name`
    (
        r"\[`([A-Z][a-zA-Z]+)`?\]\(.*?\)",
        r":class:`.\1`",
    ),  # MD object links -> :class:`name`
    (r"([^:])(`[<>/a-zA-Z0-9_\- \.,\"\']+?`)", r"\1`\2`"),  # `text` -> ``text``
    (
        r"\[([a-zA-Z/_\(\):\-\. \(\)]+)?\]\((.*)?\)",
        r"`\1 <\2>`_",
    ),  # MD links -> RT links
    (
        r"^See also:.*$",
        "See also:\n",
    ),  # See also: -> See also: \n (for nicer formatting of lists)
]
TO_PROCESS = [
    (r":meth:`[a-z][a-zA-Z]+?`", camel_case_to_snake_case),  # CamelCase -> snake_case
    (
        r"``[a-z][a-zA-Z]+?`` argument",
        camel_case_to_snake_case,
    ),  # CamelCase -> snake_case
]
LINES_TO_IGNORE_REGEXES = [
    r"---",  # Lines
    r"^\|.+\|$",  # Tables
    # r"^Config key: .*$",  # Config annotations
    # r"^Note: .*$",  # Notes
    r"^Python alias .*$",  # Python aliases
    r"^#+.*",  # Other headings
]


if __name__ == "__main__":
    with open(OUTPUT_FILE, "w") as output_file:
        output_file.write(
            """/*
    This file is autogenerated by scripts/create_python_docstrings_from_cpp_docs.py
    Do not edit it manually. Edit the Markdown files under docs/api/cpp/ directory instead and regenerate it.
*/

#ifndef __VIZDOOM_METHODS_DOCSTRINGS_H__
#define __VIZDOOM_METHODS_DOCSTRINGS_H__

namespace vizdoom {
namespace docstrings {

"""
        )

        for file in FILES_TO_PARSE:
            if "namespace" in file:
                output_file.write(f"namespace {file['namespace']} {{\n\n")

            with open(file["filepath"]) as input_file:
                lines = input_file.readlines()

            started = False
            next_docstring = ""
            for line in lines:
                # If lines match pattern, extract the function name and arguments
                match = re.match(FUNCTION_HEADER_REGEX, line)
                if match:
                    if started:
                        next_docstring = next_docstring.strip()
                        next_docstring += f'){RAW_STRING_ESCAPE_SEQ}";\n\n'  # noqa
                        output_file.write(next_docstring)

                    next_docstring = ""
                    function_name = match.group(1)
                    output_file.write(
                        f'    const char *{function_name} = R"{RAW_STRING_ESCAPE_SEQ}('
                    )
                    started = True

                elif started:
                    # Filter out some lines
                    if not any(re.match(r, line) for r in LINES_TO_IGNORE_REGEXES):
                        # Replace some patterns
                        for r in TO_REPLACE:
                            line = re.sub(r[0], r[1], line)
                        for r in TO_PROCESS:
                            line = re.sub(
                                r[0], lambda match: r[1](match.group(0)), line
                            )
                        next_docstring += line

            if started:
                output_file.write(
                    f'{next_docstring.strip()}){RAW_STRING_ESCAPE_SEQ}";\n\n'  # noqa
                )

            if "namespace" in file:
                output_file.write(f"}} // namespace {file['namespace']}\n\n")

        output_file.write(
            """
} // namespace docstrings
} // namespace vizdoom

#endif
"""
        )
