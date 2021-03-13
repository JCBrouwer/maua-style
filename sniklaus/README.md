# Optical flow implementations by @sniklaus

A collection of excellent optical flow model reproductions.
Any or all of these models can be used to calculate optical flow for video style transfers.
Each repository has its own license and terms.
Make sure to read and adhere to them as well as cite accordingly when using optical flow.

Note: the default video style setting averages the flow from all of these models.

Note: you will need to remove some shape asserts from the run.py in each repository. 
In my experience this improves speed and quality relative to resizing images to the shapes that the flow model expects.
On Linux this can be done (from the root of the repo) with:
```language=bash
find sniklaus/ -name "run.py" -exec sed -i 's/assert.*//g' {} \;
```