**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./code_images/data_visualization.png "Visualization"
[image2]: ./code_images/originalfig.png "Original Image"
[image3]: ./code_images/deformation.png "After Deformation"
[image4]: ./code_images/2_stage_ConvNet.jpg "2_stage_ConvNet"
[image5]: ./new-images/stop.jpg "Stop Sign"
[image6]: ./new-images/roadwork.jpg "Road Work"
[image7]: ./new-images/childrencrossing.jpg "Children Crossing"
[image8]: ./new-images/keepright.jpg "Keep Right"
[image9]: ./new-images/speedlimit70.png "speedlimit70"
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set.
I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the classes are distributed in training set ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data.

Based on the paper [Sermanet Lecun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I preprocessed the image data as follows:

* Convert the image to YUV space and use Y channel only
* Apply Histogram Equalization to adjust the contrast
* Normalization: center each image around its mean value and divided by its standard deviation


Next step is data augmentation using imagedatagenerator from keras library.

To avoid overfitting and yield more robust learning process, I added 20000 transformed versions of the original training set.

Here is the code for deformation:
image_datagen = ImageDataGenerator(rotation_range=15.,
                                   zoom_range=0.1,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)
                                   
Here is an example of an original image and an augmented image:

![alt text][image2] ![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

The model architecture is based on the 2-stage ConvNet as shown in the figure:

![alt text][image4]

My final model consisted of the following layers:

| Layer 1           		|     Description	        				    	| 
|:-------------------------:|:-------------------------------------------------:| 
| Input             		| inputs: 32x32x1    						    	| 
| Convolution 3x3       	| 1x1 stride, same padding, outputs: 32x32x64    	|
| RELU  					|										    		|
| Max pooling   	      	| 2x2 kernel, 2x2 stride, outputs: 16x16x64     	|

| Layer 2            		|     Description	        			    		| 
|:-------------------------:|:-------------------------------------------------:| 
| Convolution 3x3   	    | 1x1 stride, same padding, outputs: 16x16x128   	|
| RELU			     		|											    	|
| Max pooling	         	| 2x2 kernel, 2x2 stride, outputs: 8x8x128   		|

| Layer 3           		|     Description	             					| 
|:-------------------------:|:-------------------------------------------------:| 
| Convolution 3x3   	    | 1x1 stride, same padding, outputs: 8x8x256    	|
| RELU   				 	|						     						|

| Fully Connected Layer 1	|     Description	             					| 
|:-------------------------:|:-------------------------------------------------:| 
| Input             		| inputs: concatenate all the ConvNets and flatten 	| 
| Output             		| outputs: 128                                   	| 

| Fully Connected Layer 2	|     Description	             					| 
|:-------------------------:|:-------------------------------------------------:| 
| Input             		| inputs: 128                                   	| 
| Output             		| outputs: 43                                   	| 
| RELU   				 	|						     						|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Optimizer: Adam algorithm which achieves good results fast.
* Batch size: 128
* Number of Epochs: 20
* Learning rate: 0.001
* Dropout: 0.8

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The model architecture is based on the paper [Sermanet Lecun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The authors provide a 2-stage ConvNet architecture which combines representation from multiple stages in the classifier. In the case of 2 stages of features, the second stage extracts 'global' and invariant shapes and structures, while the first stage extracts 'local' motifs with more precise details [Sermanet Lecun].

My final model results were:
* training set accuracy of 
* validation set accuracy of ? 
* test set accuracy of ?

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        		    	| 
|:---------------------:|:-----------------------------------------:| 
| Stop Sign      		| Stop Sign   						    	| 
| Road Work	    		| Road Work									|
| Children Crossing	   	| Speed limit (50km/h)  	 				|
| Keep Right    	   	| Keep Right            	 				|
| Speed limit (70km/h)	| Speed limit (70km/h)     	    			|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This performance is obviously worse than the performance on the test set. However 5 images is too few for any statistical analysis.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Top 5 model predictions for image 0 (Target is 14)
   Prediction = 14 with confidence 1.00
   Prediction = 17 with confidence 0.00
   Prediction = 12 with confidence 0.00
   Prediction = 15 with confidence 0.00
   Prediction = 33 with confidence 0.00
Top 5 model predictions for image 1 (Target is 25)
   Prediction = 25 with confidence 0.98
   Prediction = 30 with confidence 0.01
   Prediction = 27 with confidence 0.00
   Prediction = 11 with confidence 0.00
   Prediction = 29 with confidence 0.00
Top 5 model predictions for image 2 (Target is 28)
   Prediction = 02 with confidence 0.20
   Prediction = 05 with confidence 0.18
   Prediction = 14 with confidence 0.17
   Prediction = 01 with confidence 0.10
   Prediction = 03 with confidence 0.08
Top 5 model predictions for image 3 (Target is 38)
   Prediction = 38 with confidence 0.97
   Prediction = 34 with confidence 0.03
   Prediction = 36 with confidence 0.00
   Prediction = 03 with confidence 0.00
   Prediction = 35 with confidence 0.00
Top 5 model predictions for image 4 (Target is 04)
   Prediction = 04 with confidence 0.96
   Prediction = 18 with confidence 0.01
   Prediction = 00 with confidence 0.01
   Prediction = 01 with confidence 0.01
   Prediction = 15 with confidence 0.00

The children crossing sign is hard to recognize due to the bad direction of the camera. The softmax distribution also shows that the model is not sure which label is correct.
