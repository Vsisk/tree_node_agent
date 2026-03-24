from __future__ import annotations

from dataclasses import dataclass


class JsonPathSyntaxError(ValueError):
    """Raised when a json_path expression has invalid syntax."""


@dataclass(frozen=True)
class FieldToken:
    name: str


@dataclass(frozen=True)
class IndexToken:
    value: int


@dataclass(frozen=True)
class JsonPath:
    tokens: tuple[FieldToken | IndexToken, ...]



def parse(json_path: str) -> JsonPath:
    """
    Parse a json_path expression like:
      $.mapping_content.children[0].children[1]

    Supported grammar (sufficient for current tree use-case):
      path := '$' ('.' field | '[' index ']')*
    """
    if not isinstance(json_path, str) or not json_path:
        raise JsonPathSyntaxError("json_path must be a non-empty string")
    if json_path[0] != "$":
        raise JsonPathSyntaxError("json_path must start with '$'")

    i = 1
    n = len(json_path)
    tokens: list[FieldToken | IndexToken] = []

    while i < n:
        ch = json_path[i]
        if ch == ".":
            i += 1
            if i >= n:
                raise JsonPathSyntaxError("field name expected after '.'")
            start = i
            while i < n and (json_path[i].isalnum() or json_path[i] == "_"):
                i += 1
            if start == i:
                raise JsonPathSyntaxError("invalid field token")
            tokens.append(FieldToken(json_path[start:i]))
            continue

        if ch == "[":
            i += 1
            start = i
            while i < n and json_path[i].isdigit():
                i += 1
            if start == i:
                raise JsonPathSyntaxError("array index expected inside []")
            if i >= n or json_path[i] != "]":
                raise JsonPathSyntaxError("missing closing ']' for array index")
            tokens.append(IndexToken(int(json_path[start:i])))
            i += 1
            continue

        raise JsonPathSyntaxError(f"unexpected character at position {i}: {ch}")

    return JsonPath(tokens=tuple(tokens))
