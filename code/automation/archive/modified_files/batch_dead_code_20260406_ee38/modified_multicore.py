aug = BackgroundAugmenter(..., threaded=True)
try:
    # Use aug in your loop
    for batch in aug:
        # process batch
finally:
    aug.terminate()
    del aug