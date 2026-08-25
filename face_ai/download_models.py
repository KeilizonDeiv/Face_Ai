"""
One-time setup script: downloads the ONNX model weights used for
face detection (YuNet) and face recognition (SFace) from the OpenCV Zoo.

Run once after installing requirements.txt:
    python download_models.py
"""

import hashlib
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# (filename, download URL, expected sha256)
MODELS = [
    (
        'face_detection_yunet_2023mar.onnx',
        'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
        'models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    ),
    (
        'face_recognition_sface_2021dec.onnx',
        'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
        'models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
        '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
    ),
]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(filename, url, expected_sha256):
    dest = os.path.join(MODELS_DIR, filename)

    if os.path.exists(dest) and sha256_of(dest) == expected_sha256:
        print(f'  already have {filename}')
        return

    print(f'  downloading {filename} ...')
    tmp_path = dest + '.tmp'
    urllib.request.urlretrieve(url, tmp_path)

    actual_sha256 = sha256_of(tmp_path)
    if actual_sha256 != expected_sha256:
        os.remove(tmp_path)
        raise RuntimeError(
            f'checksum mismatch for {filename}: '
            f'expected {expected_sha256}, got {actual_sha256}'
        )

    os.replace(tmp_path, dest)
    print(f'  ✓ {filename}')


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print('Downloading face detection/recognition models into models/ ...')
    for filename, url, expected_sha256 in MODELS:
        download(filename, url, expected_sha256)
    print('Done. Models are stored locally in models/ and used fully offline from now on.')


if __name__ == '__main__':
    main()
