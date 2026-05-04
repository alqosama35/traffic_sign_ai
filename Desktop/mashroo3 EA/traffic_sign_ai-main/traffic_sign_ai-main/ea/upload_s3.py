import boto3
import os
import pathlib


def upload_log(local_path: str, run_id: str) -> str:
    """
    Upload a single JSON log to S3.
    Requires S3_BUCKET environment variable to be set.
    """
    bucket = os.environ['S3_BUCKET']
    key = f"ea-experiments/{run_id}.json"
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)
    url = f"s3://{bucket}/{key}"
    print(f"Uploaded: {url}")
    return url


def upload_all_logs(results_dir='ea/results/') -> list:
    """
    Upload all JSON logs in ea/results/ to S3.
    """
    uploaded = []
    for path in pathlib.Path(results_dir).glob('*.json'):
        run_id = path.stem
        url = upload_log(str(path), run_id)
        uploaded.append(url)
    return uploaded


def list_s3_logs():
    """
    List all logs currently in the S3 bucket.
    """
    bucket = os.environ['S3_BUCKET']
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix='ea-experiments/')
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"  {obj['Key']} ({obj['Size']} bytes)")
    else:
        print("  No logs found in S3.")


if __name__ == '__main__':
    print("Uploading all EA experiment logs to S3...")
    urls = upload_all_logs()
    print(f"\nUploaded {len(urls)} logs.")
    print("\nVerifying S3 contents:")
    list_s3_logs()