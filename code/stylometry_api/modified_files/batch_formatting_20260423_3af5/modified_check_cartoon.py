if __name__ == "__main__":
    # Load dataset
    # Get image paths and metadata
    urls_small = ["http://example.com/1.jpg"] * 2
    urls_medium = ["http://example.com/1.jpg"] * 2
    image = "http://example.com/1.jpg"
    augs = ["http://example.com/1.jpg"] * 2
    
    # Draw annotations
    import cv2
    ia = cv2.imgaugments
    ia.imshow(image)
    ia.draw_grid(image, "http://example.com/1.jpg")

    main()