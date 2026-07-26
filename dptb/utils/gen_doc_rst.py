"""Generate compact input-parameter reference pages."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dargs.dargs import Argument

from dptb.utils.argcheck import (
    common_options,
    data_options,
    embedding,
    prediction,
    run_options,
    slem,
    train_options,
)


DOC_PATH = Path("docs/input_params")


def gen_doc_list(arglist, *, make_anchor=False, make_link=False):
    if isinstance(arglist, Argument):
        arglist = [arglist]
    if not isinstance(arglist, list):
        raise TypeError("arglist must be an Argument or list of Arguments")
    generated = "\n\n".join(
        arg.gen_doc(make_anchor=make_anchor, make_link=make_link)
        for arg in arglist
    )
    return "\n".join(line.rstrip() for line in generated.splitlines())


def _heading(title: str, marker: str = "-") -> str:
    return f"{title}\n{marker * len(title)}"


def _compact_model_doc() -> str:
    """Document shared embedding fields once instead of once per alias."""

    embedding_variant = embedding()
    prediction_variant = prediction()
    base_fields = slem()
    base_names = {field.name for field in base_fields}

    method_extra_names = {}
    extension_fields = {}
    for method, choice in embedding_variant.choice_dict.items():
        extras = [
            field for field in choice.sub_fields.values()
            if field.name not in base_names
        ]
        method_extra_names[method] = [field.name for field in extras]
        for field in extras:
            extension_fields.setdefault(field.name, field)

    lines = [
        _heading("Model Options", "="),
        (
            "``0726-light`` keeps one embedding-plus-prediction model family. "
            "The strict runtime schema in ``dptb.utils.argcheck`` remains the "
            "source of truth. This page intentionally documents shared fields "
            "once so aliases do not multiply the same schema."
        ),
        _heading("Supported embedding methods"),
        "\n".join(
            f"* ``{method}``"
            for method in embedding_variant.choice_dict
        ),
        _heading("Shared embedding options"),
        gen_doc_list(
            Argument(
                "shared_embedding_options",
                dict,
                sub_fields=base_fields,
                sub_variants=[],
                optional=True,
            )
        ),
        _heading("Method-specific embedding keys"),
    ]

    for method, names in method_extra_names.items():
        rendered = ", ".join(f"``{name}``" for name in names)
        lines.append(
            f"* ``{method}``: {rendered if rendered else 'shared options only'}"
        )

    if extension_fields:
        lines.extend(
            [
                _heading("Embedding extension option reference"),
                gen_doc_list(
                    Argument(
                        "embedding_extension_options",
                        dict,
                        sub_fields=list(extension_fields.values()),
                        sub_variants=[],
                        optional=True,
                    )
                ),
            ]
        )

    lines.extend(
        [
            _heading("Prediction methods"),
            "\n".join(
                f"* ``{method}``"
                for method in prediction_variant.choice_dict
            ),
        ]
    )
    for method, choice in prediction_variant.choice_dict.items():
        lines.extend(
            [
                _heading(f"{method} prediction options", "~"),
                gen_doc_list(choice),
            ]
        )
    return "\n\n".join(lines)


def _first_line(text: str) -> str:
    return " ".join(str(text or "").strip().split())[:240].rstrip()


def _compact_train_doc() -> str:
    """Render the large, fast-moving training schema as a navigable index."""

    schema = train_options()
    lines = [
        _heading("Train Options", "="),
        (
            "Training, flow, distributed execution, monitoring, checkpoint, "
            "and loss controls. The strict schema in "
            "``dptb.utils.argcheck.train_options`` is authoritative; this "
            "compact page avoids duplicating thousands of generated lines."
        ),
    ]
    for name, field in schema.sub_fields.items():
        summary = _first_line(field.doc)
        default = ""
        if field.optional:
            default = f"; default ``{field.default!r}``"
        label = f"* ``{name}``{default}"
        if summary:
            label += f" — {summary}"
        lines.append(label)

        child_names = list(field.sub_fields)
        if child_names:
            lines.append(
                "  Nested keys: "
                + ", ".join(f"``{child}``" for child in child_names)
            )
        for variant in field.sub_variants.values():
            choices = ", ".join(
                f"``{choice}``" for choice in variant.choice_dict
            )
            lines.append(
                f"  ``{variant.flag_name}`` choices: {choices}"
            )
    return "\n\n".join(lines)


def _write_page(filename: str, title: str, options) -> None:
    content = "\n".join(
        [
            _heading(title, "="),
            gen_doc_list(options),
            "",
        ]
    )
    (DOC_PATH / filename).write_text(content, encoding="utf-8")


def main() -> None:
    DOC_PATH.mkdir(parents=True, exist_ok=True)
    index = "\n".join(
        [
            _heading("Full Input Parameters", "="),
            "",
            ".. toctree::",
            "   :maxdepth: 2",
            "",
            "   common_options",
            "   train_options",
            "   model_options",
            "   data_options",
            "   run_options",
            "",
        ]
    )
    (DOC_PATH / "index.rst").write_text(index, encoding="utf-8")
    _write_page("common_options.rst", "Common Options", common_options())
    (DOC_PATH / "train_options.rst").write_text(
        _compact_train_doc() + "\n",
        encoding="utf-8",
    )
    (DOC_PATH / "model_options.rst").write_text(
        _compact_model_doc() + "\n",
        encoding="utf-8",
    )
    _write_page("data_options.rst", "Data Options", data_options())
    _write_page("run_options.rst", "Run Options", run_options())


if __name__ == "__main__":
    main()
