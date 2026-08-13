from habit import Spec, parse_feature_expression

expression = 'concat(raw("T1"), local_entropy("T2", kernel_size=3))'
parsed = parse_feature_expression(expression)
structured = Spec(
    "concat",
    {
        "children": [
            {"name": "raw", "params": {"modality": "T1"}},
            {"name": "local_entropy", "params": {"modality": "T2", "kernel_size": 3}},
        ],
    },
)
assert parsed.fingerprint() == structured.fingerprint()
print(parsed.fingerprint())
print(structured.fingerprint())