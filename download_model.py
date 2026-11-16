"""Download public BERT weights with bounded timeouts and integrity verification."""
import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from common import ROOT

URLS = ['https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin',
        'https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin']
EXPECTED_SHA256 = '097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a'


def probe(url):
    start = time.monotonic()
    try:
        with requests.get(url, headers={'Range': 'bytes=0-1023'}, stream=True, timeout=(10, 20)) as response:
            response.raise_for_status()
            chunk = next(response.iter_content(1024))
            print(url, response.status_code, len(chunk), round(time.monotonic() - start, 1), flush=True)
            return url
    except Exception as exc:
        print(url, type(exc).__name__, str(exc)[:150], flush=True)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    with ThreadPoolExecutor(2) as pool:
        working = [url for url in pool.map(probe, URLS) if url]
    if args.probe:
        return
    destination = ROOT / 'models/bert-base-uncased/pytorch_model.bin'
    for url in working:
        tmp = destination.with_suffix('.download')
        try:
            with requests.get(url, stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                size, start = 0, time.monotonic()
                digest = hashlib.sha256()
                with tmp.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        if size % (20 * 1024 * 1024) < 1024 * 1024:
                            print('Downloaded %.1f MB in %.1f s' % (size / 1e6, time.monotonic() - start), flush=True)
            if digest.hexdigest() != EXPECTED_SHA256:
                raise ValueError('BERT checksum mismatch: ' + digest.hexdigest())
            tmp.replace(destination)
            print('Verified BERT SHA256', digest.hexdigest(), flush=True)
            return
        except Exception as exc:
            print('Download failed', url, str(exc), flush=True)
    raise RuntimeError('No successful BERT download')


if __name__ == '__main__':
    main()
