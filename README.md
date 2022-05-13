# Fluorescent Fiber Segmentation Web App

The application takes as input images of fluorescent fibers in TIFF or PNG format. A user can enter a single file or multiple at a time. After the application runs, it will load a download screen where the user can download the fiber tracings in SVG format. The SVG files can then be aligned with their original image in a vector graphics software, like Adobe Illustrator, to observe the detected cells.

Please refer to my [project blog](https://sites.google.com/view/project-blogs/blogs/from-threshold-to-fibers?authuser=0) for implementation details.

## Usage

Once you download the repo and install the dependencies, simply run `python app.py` and a browser window will open with the app. 

## Resources

* https://github.com/LingDong-/skeleton-tracing
* https://github.com/alexarnal/nissl_mapping
