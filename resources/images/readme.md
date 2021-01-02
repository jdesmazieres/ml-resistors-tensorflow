This directory contains resistor images archives

* resistors-176x64.zip: raw images with all a 176x64 size. Each image appears 4 times (4 positions) to increase dataset size and so the accuracy of the model. This could have been achieved using dataset augmentation during training
* resistors-176x64-preprocessed.zip: previous image dataset with image preprocessing applied on (shadow remove + gaussian blur)

To be able to run the scripts, these archives should be unzipped in the current directory (keeping the directory structure) 
