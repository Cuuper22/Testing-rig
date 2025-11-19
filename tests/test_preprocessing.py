import pytest

from ml_project.preprocessing import (
    build_tokenizer,
    decode_tokens,
    encode_tokens,
    scale_features,
)


def test_scale_features_learns_parameters():
    features = [[1.0, 2.0], [3.0, 4.0]]
    scaled, params = scale_features(features)

    # Mean should be approximately zero and std approximately one.
    for column in zip(*scaled):
        assert pytest.approx(sum(column) / len(column), abs=1e-6) == 0.0
        variance = sum(value**2 for value in column) / len(column)
        assert pytest.approx(variance, abs=1e-6) == 1.0

    scaled_again, _ = scale_features(features, params=params)
    assert scaled == scaled_again

    new_features = [[value + 10 for value in row] for row in features]
    transformed, _ = scale_features(new_features, params=params)
    for row in transformed:
        for value in row:
            assert value >= 0  # shifted data should move away from the mean


def test_scale_features_raises_for_invalid_shape():
    with pytest.raises(ValueError):
        scale_features([])


def test_tokenizer_round_trip():
    vocab = ["hello", "world", "hello"]
    token_to_id = build_tokenizer(vocab)
    encoded = encode_tokens(["hello", "world"], token_to_id)
    decoded = decode_tokens(encoded, token_to_id)
    assert decoded == ["hello", "world"]


def test_encode_unknown_token_raises():
    token_to_id = {"hello": 0}
    with pytest.raises(KeyError):
        encode_tokens(["unknown"], token_to_id)
