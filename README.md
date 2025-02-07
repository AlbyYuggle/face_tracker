# Dependencies

To install all dependencies in a virtual environment use 

```
python3 -m venv face_tracker_env
source face_tracker_env/bin/activate
pip install -r requirements.txt
```
# Running the code

To run the code, run this example:

```
python run.py -i input/test.mp4 -o output/test  -r input/reference_self.jpg
```

Here, -i is the input path to the video, -o is the output directory, and -r is the input reference image, and these can be replaced with your own data.

# Discussion

## Algorithm

The algorithm for face tracking combines a pretrained face detection/recognition model with 2D kalman filter tracking. The pretrained face recognition model is good for recognizing the correct face, but often has false negatives. To solve this issue, in frames marked as false negative, a kalman filter tracker is used to predict where the face would be if there was a face, then a face detector (empirically has less false negatives) checks if there is a face in the predicted location with IOU metric. 

## Example Cases

I have used some self-collected example cases in the input/output folders. Each test case corresponds to the subfolder with the same name in the output folder.

### test

```
python run.py -i input/test.mp4 -o output/test  -r input/reference_self.jpg
```

This test case tests tracking for one person (me) with a reference image taken at a different time/location/angle. I move around and enter and exit the camera view, including at different scales.

### test_false

```
python run.py -i input/test.mp4 -o output/test_false  -r input/reference_false.jpg
```

This test case tests the same tracking as above with an incorrect reference image.

### test_occ

```
python run.py -i input/test_occ.mp4 -o output/test_occ  -r input/reference_self.jpg
```

This test case tests the occlusion case for one person (me) with a reference image taken at a different time/location/angle.

### yt

```
python run.py -i input/yt.mp4 -o output/yt  -r input/reference_yt.jpg
```

This test case also tests tracking for one moving person with a reference image taken at a different time/location/angle. This test case tests the algorithms ability for "in the wild" examples, i.e. sourced from YouTube.

### yt2

```
python run.py -i input/yt2.mp4 -o output/yt2  -r input/reference_yt2.jpg
```

This test case also tests tracking for one person with a reference image taken at a different time/location/angle. This test case tests the algorithms ability to handle scene cuts and color shifts (grayscale), and is sourced from YouTube.


### yt3


```
python run.py -i input/yt3.mp4 -o output/yt3  -r input/reference_yt3.jpg
```

The most comprehensive test case, this test case features multiple people, many cuts, and many different scales, and is sourced from YouTube.

## Limitations

Sometimes the clips should be continuous but may be split into two clips or more clips, which is caused by the facial recognition module predicting a false negative and a bad prediction from the kalman filter tracker. This can be alleviated by further tuning paramaters such as kalman filter timestep/noise parameters and IOU threshold. Despite this, the results are generally good for the testcases I have tried. 

## Credits

Data was collected from videos for the following youtube channels:

https://www.youtube.com/@penguinz0

https://www.youtube.com/channel/UC9GtSLeksfK4yuJ_g1lgQbg

https://www.youtube.com/@Ruxin34

This code utilized pretrained models from:

https://github.com/serengil/deepface/tree/master

```
@article{serengil2024lightface,
  title     = {A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules},
  author    = {Serengil, Sefik and Ozpinar, Alper},
  journal   = {Journal of Information Technologies},
  volume    = {17},
  number    = {2},
  pages     = {95-107},
  year      = {2024},
  doi       = {10.17671/gazibtd.1399077},
  url       = {https://dergipark.org.tr/en/pub/gazibtd/issue/84331/1399077},
  publisher = {Gazi University}
}
```