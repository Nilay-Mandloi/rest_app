import pytest


def test_resolve_and_load_via_stable_channel(cache):
    loaded = cache.resolve_and_load(
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        channel="stable",
    )
    assert loaded.version_id == "v1"
    assert loaded.manifest.model_name == "sentiment_analysis"
    assert hasattr(loaded.obj, "predict")


def test_cache_hit_returns_same_object(cache):
    first = cache.resolve_and_load(
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        channel="stable",
    )
    second = cache.resolve_and_load(
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        version_id="v1",
    )
    assert first is second


def test_explicit_version_skips_pointer(cache, s3_world, publish_artifacts, bucket_name):
    # Publish v2 without updating stable.json — stable still points at v1.
    publish_artifacts(
        s3_world,
        bucket_name,
        "product_dq",
        "sentiment_analysis",
        2,
        feature_columns=["a", "b"],
        channel="_unused",
    )
    loaded = cache.resolve_and_load(
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        version_id="v2",
    )
    assert loaded.version_id == "v2"


def test_checksum_mismatch_rejected(cache, s3_world, publish_artifacts, bucket_name):
    publish_artifacts(
        s3_world,
        bucket_name,
        "product_dq",
        "sentiment_analysis",
        5,
        bad_checksum=True,
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        cache.resolve_and_load(
            category="mlops",
            project="product_dq",
            model_name="sentiment_analysis",
            version_id="v5",
        )


def test_tenant_mismatch_rejected(cache, s3_world, publish_artifacts, bucket_name):
    publish_artifacts(
        s3_world,
        bucket_name,
        "product_dq",
        "sentiment_analysis",
        3,
        category="other_cat",
    )
    with pytest.raises(RuntimeError, match="manifest tenant mismatch"):
        cache.resolve_and_load(
            category="mlops",
            project="product_dq",
            model_name="sentiment_analysis",
            version_id="v3",
        )


def test_lru_evicts_oldest_when_full(settings, s3_world, publish_artifacts, bucket_name):
    from dataclasses import replace

    from rest_app.adapters.s3_store import S3ReadStore
    from rest_app.loader import ModelCache

    tiny = replace(settings, cache_max_entries=2)
    c = ModelCache(tiny, store=S3ReadStore(client=s3_world))

    for v in (1, 2, 3):
        publish_artifacts(
            s3_world,
            bucket_name,
            "product_dq",
            "sentiment_analysis",
            v,
            feature_columns=["a", "b"],
            channel="_v" + str(v),
        )
        c.resolve_and_load(
            category="mlops",
            project="product_dq",
            model_name="sentiment_analysis",
            version_id=f"v{v}",
        )

    entries = c.list_entries()
    versions = sorted(e["version_id"] for e in entries)
    assert len(entries) == 2
    assert versions == ["v2", "v3"]


def test_missing_pointer_raises_file_not_found(cache):
    with pytest.raises(FileNotFoundError, match="pointer not found"):
        cache.resolve_and_load(
            category="mlops",
            project="product_dq",
            model_name="sentiment_analysis",
            channel="canary",  # not published
        )
