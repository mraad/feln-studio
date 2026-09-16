"""Catalog-aware FELN compilation and the gold record every backend is judged against.

`Schema` is feln-lora's `src.feln_data.Schema` (validate/compile/context only).
ponytail: duplicated because feln-lora is not an installable package; import it once it is.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import sqlglot
import sqlglot.errors
from feln import FELN, parse_relation, to_meters
from sqlglot import exp

LIKE_TYPES = (exp.Like, exp.ILike)
COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, *LIKE_TYPES)


def parse_predicate(clause: str):
    """One SQL predicate as an AST; malformed SQL is a ValueError like every other rejection."""
    try:
        expressions = sqlglot.parse(clause)
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"Invalid SQL predicate: {clause}") from exc
    if len(expressions) != 1 or not isinstance(
        expressions[0], (exp.Predicate, exp.Connector, exp.Not, exp.Paren)
    ):
        raise ValueError(f"Expected one SQL predicate: {clause}")
    return expressions[0]


class Schema:
    def __init__(self, path: Path):
        self.path = path
        self.raw = json.loads(path.read_text())
        # Standalone tables carry no geometry, so FELN can never target them.
        self.layers = {
            layer["name"]: layer for layer in self.raw["layers"] if layer.get("stype") != "Table"
        }
        self.columns = {
            name: {c["name"].lower(): c for c in layer["columns"]}
            for name, layer in self.layers.items()
        }

    def validate(self, meta: dict) -> FELN:
        feln = FELN.model_validate(meta)
        for rel in feln.relations:
            relation = parse_relation(rel)  # feln accepts any unit; the SQL compiler does not
            if relation.unit and (
                relation.distance < 0 or to_meters(relation.distance, relation.unit) is None
            ):
                raise ValueError(f"Unsupported distance relation: {rel}")
        for name, clause in zip(feln.layers, feln.where):
            if name not in self.layers:
                raise ValueError(f"Unknown layer {name}")
            if not clause:
                continue
            tree = parse_predicate(clause)
            if any(
                isinstance(node, exp.Func) and not isinstance(node, (exp.Cast, exp.And, exp.Or))
                for node in tree.walk()
            ):
                raise ValueError(
                    "SQL functions other than type casts are outside the supported filter grammar"
                )
            for column in tree.find_all(exp.Column):
                if column.table or column.name.lower() not in self.columns[name]:
                    raise ValueError(f"Unknown field {name}.{column.sql()}")
            # Reject hallucinated enum codes without treating sampled string values
            # as exhaustive domains: new facilities/operators are legitimate.
            checks = []
            for predicate in tree.walk():
                if isinstance(predicate, COMPARISONS):
                    checks.append((predicate, predicate.expression))
                elif isinstance(predicate, exp.In):
                    checks.extend((predicate, value) for value in predicate.expressions)
                elif isinstance(predicate, exp.Between):
                    checks.extend((predicate, predicate.args[key]) for key in ("low", "high"))
            for predicate, value in checks:
                column = predicate.this
                if isinstance(column, exp.Column):
                    spec = self.columns[name][column.name.lower()]
                    if isinstance(value, exp.Cast):
                        value = value.this
                    if (
                        spec["keyval"]
                        and spec["dtype"] != "String"
                        and value.sql() not in spec["keyval"]
                    ):
                        raise ValueError(f"Invalid enum code: {predicate.sql()}")
                    if isinstance(value, exp.Literal):
                        string_type = spec["dtype"] in {"String", "Date"}
                        if value.is_string != string_type:
                            raise ValueError(f"Wrong literal type: {predicate.sql()}")
                        if spec["keyval"] and string_type:
                            known = set(spec["keyval"]) | set(spec["values"])
                            if value.this and value.this not in known:
                                raise ValueError(f"Unknown coded value: {predicate.sql()}")
        return feln

    def compile(self, meta: dict) -> dict:
        """Emit schema-typed SQL, preserving predicates and spatial joins.

        Resolve exact enum labels/codes, explicit uppercase policies, and uniquely
        matched spelling in the supplied value catalog. Unknown names are never
        invented. Keep raw-model accuracy separate from compiled-output accuracy.
        """
        result = FELN.model_validate(meta).model_dump()
        names = {name.casefold(): name for name in self.layers}
        types = {
            "SmallInteger": "SMALLINT",
            "Integer": "INTEGER",
            "Double": "DOUBLE PRECISION",
        }
        for i, name in enumerate(result["layers"]):
            if name.casefold() not in names:
                raise ValueError(f"Unknown layer {name}")
            name = result["layers"][i] = names[name.casefold()]
            if not result["where"][i]:
                continue
            tree = parse_predicate(result["where"][i])
            for column in tree.find_all(exp.Column):
                if column.table or column.name.lower() not in self.columns[name]:
                    raise ValueError(f"Unknown field {name}.{column.sql()}")
                column.set(
                    "this",
                    exp.to_identifier(self.columns[name][column.name.lower()]["name"], quoted=True),
                )
            for predicate in list(tree.walk()):
                if not isinstance(predicate, (*COMPARISONS, exp.In, exp.Between)):
                    continue
                if not isinstance(predicate.this, exp.Column):
                    continue
                spec = self.columns[name][predicate.this.name.lower()]
                if isinstance(predicate, exp.In):
                    values = list(predicate.expressions)
                elif isinstance(predicate, exp.Between):
                    values = [predicate.args["low"], predicate.args["high"]]
                else:
                    values = [predicate.expression]
                for value in values:
                    # Preserve existing casts: replacing a cast could change truncation.
                    if isinstance(value, exp.Cast):
                        continue
                    literal_node = value.this if isinstance(value, exp.Neg) else value
                    if not isinstance(literal_node, exp.Literal):
                        continue
                    text = literal_node.this if literal_node.is_string else value.sql()
                    mapped = {
                        key
                        for key, label in spec["keyval"].items()
                        if str(text).casefold() in (key.casefold(), label.casefold())
                    }
                    if len(mapped) == 1:
                        text = mapped.pop()
                    if spec["dtype"] in types:
                        try:
                            number = Decimal(text)
                        except InvalidOperation as exc:
                            raise ValueError(f"Invalid numeric value: {text}") from exc
                        if not number.is_finite() or abs(number.adjusted()) > 308:
                            raise ValueError("Numeric value is outside supported range")
                        if spec["dtype"] != "Double":
                            bits = 16 if spec["dtype"] == "SmallInteger" else 32
                            if number != number.to_integral_value() or not -(
                                2 ** (bits - 1)
                            ) <= number < 2 ** (bits - 1):
                                raise ValueError(
                                    "Integer value is fractional or overflows its schema type"
                                )
                        text = format(number, "f")
                        text = text.rstrip("0").rstrip(".") if "." in text else text
                        value.replace(
                            exp.Cast(
                                this=sqlglot.parse_one(text),
                                to=exp.DataType.build(types[spec["dtype"]]),
                            )
                        )
                    elif literal_node.is_string:
                        if any("uppercase the compared values" in hint for hint in spec["hints"]):
                            text = text.upper()
                        elif not spec["keyval"]:
                            needle = text.strip("%") if isinstance(predicate, LIKE_TYPES) else text
                            if needle and not any(c in needle for c in "%_"):
                                matches = {
                                    match.group()
                                    for candidate in spec["values"]
                                    for match in re.finditer(
                                        re.escape(needle), str(candidate), re.IGNORECASE
                                    )
                                    if isinstance(predicate, LIKE_TYPES)
                                    or match.group().casefold() == str(candidate).casefold()
                                }
                                if len(matches) == 1:
                                    text = text.replace(needle, matches.pop())
                        value.replace(exp.Literal.string(text))
            result["where"][i] = tree.sql(dialect="postgres")
        self.validate(result)
        return result


def key(text: str) -> str:
    return " ".join(text.casefold().split())


class Gold:
    """Recorded question → FELN, matched on whitespace/case-insensitive text."""

    def __init__(self, paths: list[Path]):
        self.records: dict[str, tuple[str, dict]] = {}
        for path in paths:
            for record in json.loads(path.read_text()):
                self.records[key(record["text"])] = (record["text"], record["meta"])

    def questions(self) -> list[str]:
        return sorted((text for text, _ in self.records.values()), key=lambda t: (len(t), t))

    def lookup(self, query: str) -> dict | None:
        found = self.records.get(key(query))
        return found[1] if found else None


def judge(schema: Schema, feln: dict, gold: dict | None) -> tuple[dict | None, bool | None]:
    """(compiled gold, strict match) or (None, None) when the question is unrecorded."""
    if gold is None:
        return None, None
    expected = schema.compile(gold)
    return expected, FELN.model_validate(feln).same(FELN.model_validate(expected))
