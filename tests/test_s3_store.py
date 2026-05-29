from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from rest_app.adapters.s3_store import S3ReadStore


@mock_aws
def test_get_json_returns_none_for_missing_key():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ReadStore(client=client)
    assert store.get_json("test-bucket", "no/such/key.json") is None


@mock_aws
def test_get_json_raises_valueerror_on_corrupt_payload():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    client.put_object(Bucket="test-bucket", Key="bad.json", Body=b"{this is not valid json")
    store = S3ReadStore(client=client)
    with pytest.raises(ValueError, match="is not valid JSON"):
        store.get_json("test-bucket", "bad.json")
