#!/usr/bin/env python3
"""Download sample pet images for demo purposes."""

import requests
from pathlib import Path
import time

# Sample pet image URLs from Unsplash (free to use)
urls = {
    'dog_portrait.jpg': 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800',
    'cat_sitting.jpg': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800',
    'dog_running.jpg': 'https://images.unsplash.com/photo-1558788353-f76d92427f16?w=800',
    'multiple_dogs.jpg': 'https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=800',
    'cat_closeup.jpg': 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800',
}

output_dir = Path('data/raw')
output_dir.mkdir(parents=True, exist_ok=True)

downloaded = []

print("Downloading sample pet images...")
for filename, url in urls.items():
    try:
        print(f'  Downloading {filename}...', end=' ', flush=True)
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(output_dir / filename, 'wb') as f:
                f.write(response.content)
            downloaded.append(filename)
            print('✓')
        else:
            print(f'✗ (status {response.status_code})')
        time.sleep(0.5)
    except Exception as e:
        print(f'✗ ({e})')

print(f'\n✅ Downloaded {len(downloaded)}/{len(urls)} images')
for img in downloaded:
    print(f'   - {img}')
