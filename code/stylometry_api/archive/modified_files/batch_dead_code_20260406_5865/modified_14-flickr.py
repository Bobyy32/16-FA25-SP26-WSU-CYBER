#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flickr Image Downloader
Fetches and saves images from Flickr with caching support.
"""

import os
import sys

# Add parent directories to path for pattern.web import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from pattern.web import Flickr, RELEVANCY, MEDIUM, SMALL
    from pattern.web import extension as file_extension
except ImportError:
    print("Error: pattern.web library not found. Install with:")
    print("pip install pattern")
    sys.exit(1)

def fetch_flickr_images(query, max_results=5, cache=True):
    """Search and retrieve images from Flickr."""
    engine = Flickr(license=None)
    results = engine.search(
        query,
        size=MEDIUM,
        sort=RELEVANCY,
        cached=cache
    )
    return results

def download_and_save(first_image, query):
    """Download and save the first matching image."""
    if not first_image:
        print("No images found.")
        return None
    
    data = first_image.download()
    safe_query = query.replace(" ", "_")
    path = safe_query + file_extension(first_image.url)
    
    # Write image file
    with open(path, "wb") as f:
        f.write(data)
    
    return path

def main():
    query = "duracell bunny"
    print(f"Searching Flickr for: '{query}'")
    
    results = fetch_flickr_images(query, cached=False)
    
    # Display results
    for img in results:
        print(f"Author: {img.author}")
        print(f"Text: {img.text}")
        print("-" * 40)
    
    # Download first image
    if results:
        img = results[0]
        path = download_and_save(img, query)
        if path:
            print(f"Downloaded: {img.url}")
            print(f"Saved as: {path}")

if __name__ == "__main__":
    main()