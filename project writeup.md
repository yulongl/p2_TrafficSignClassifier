# **Traffic Sign Recognition** 

## Writeup

### Yulong Li

  

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize and analyze feature maps
* Summarize the results with a written report

 
---
## Rubric Points
This writeup will include all the [Rubric Points](https://review.udacity.com/#!/rubrics/481/view).  
Here is a link to my [project code](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

---
### Dataset Exploration
#### Dataset Summary
The provided German Traffic Sign Dataset consist 34799 traning images, 4410 validation images and 12630 testing images, which contain 43 different classes. All images are 32x32 3 channel RGB. The data are provided in pickle format, which contains four pairs of key and value: feature (raw pixel data of images), label, image size and traffic sign object coordinates.

#### Exploratory Visualization
Below are three examples from the training dataset:  
![2](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/2.png)
![12](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/12.png)
![26](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/26.png)  

Below are histograms for different classes in training, validation and testing data set:  
![trainhist](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/trainhist.png)
![validhist](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/validhist.png)
![testhist](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/testhist.png)  

---

### Design and Test a Model Architecture

#### Preprocessing

I decided not to convert the images to grayscale because I think color information is useful for traffic sign classification. For example the 'ahead only' sign is blue and the 'stop' sign is red. I want the CNN also learning the color information.  

I normalized the image data because it can help to avoid very big positive or negative values after the convolution and it's easier for optimizer tp work.  

I also augument the images using imgaug package.  

```
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Crop(px=(0, 6)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Add((-10, 10)), 
    iaa.Sharpen(alpha = 1.00) 
])
```

Below are afew examples of augumented images:  

![2_aug](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/2_aug.png)
![9_aug](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/9_aug.png)
![35_aug](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/35_aug.png)  


The augumented images were added to the original dataset so the number of training iamges became 69598.  


#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3x6     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| Sigmoid					|												|
| Convolution 5x5x6x16	    | 1x1 stride, VALID padding, outputs 24x24x16	|
| Sigmoid					|												|
| Convolution 5x5x16x26	    | 1x1 stride, VALID padding, outputs 20x20x26	|
| Sigmoid					|												|
| Convolution 5x5x26x36	    | 1x1 stride, VALID padding, outputs 16x16x36	|
| Sigmoid					|												|
| Convolution 5x5x36x36	    | 1x1 stride, VALID padding, outputs 12x12x36	|
| Sigmoid					|												|
| Maxpooling 2x2				  	|	2x2	stride, SAME padding, outputs 6x6x36		|
| Fully connected	1296	| Outputs 666        									|
| RELU					|												|
| Dropout				|	0.5						  	    |
| Fully connected	666 	| Outputs 396        									|
| RELU					|												|
| Dropout				|	0.5						  	 	  |
| Fully connected	396	| Outputs 128          									|
| RELU					|												|
| Dropout				|	0.5						  		  |
| Fully connected	128	| Outputs 43          									|
| RELU					|												|
| Softmax				|      									|
| Cross entropy	|      									|
| Back prop			|      									|



#### Model Training

To train the model, I used the TensorFlow AdamOptimizer.  
```
EPOCHS = 100
BATCH_SIZE = 128
mu = 0
sigma = 0.09
rate = 0.0003
dropout = 0.5
```

#### Solution Approach

First I tried the LeNet architecture from previous class but the result was below the 93% requirement. So next I tried another famous architecture AlexNet but a simplified version and achived 94%~95% validation accuracy.  

However, the image size in the provided dataset is only 32x32x3. And the images are very vague. We already lost a lot of information by downsizing the original images. And the over use of maxpooling will even lose more pixels and more information contained in those pixels. So eventually, I used convolution with VALID padding to reduce feature map size and only used maxpooling once after 5 convolution layers. After that I got a relatively large fully connected layer. To accelarate training and to prevent overfitting, I added the dropout function after each fully connected layer.

Another approach I've made is, I used sigmoid as the activation function instead of RELU. Because when I was using RELU, after step 4 visualization of the feature maps, I found too many completely black feature maps, which means for these maps, all the logits were negative, which means all those filters were dead filters. I even tried some very small learning rates and small sigma for weight initialization but still getting almost 50% of dead filters. **But the weird thing is, even if 80% of the filters were dead filters, the system still gave out a 98% validation accuracy.**  


My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.974
* test set accuracy of 0.966

 
---
### Test a Model on New Images

#### Acquiring New Images

I think five images is not enough to evaluate the system so I got 11. Here are they:  

![2](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/2.png)
![3](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/3.png)
![7](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/7.png)
![7_2](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/7_2.png)
![14](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/14.png)
![17](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/17.png)
![23](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/23.png)
![25](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/25.png)
![28](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/28.png)
![31](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/31.png)
![35](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/image_from_google/35.png)


The 'slippery road' sign image probably is the most difficault to classify because it's dark and the pattern is complicated. And the 'child crossing' and 'road work' also have complicated patterns. The '100 speed limit' is a little tricky because it's probably not a Germany sign - the read circle is thinner.  


#### Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)      		| Speed limit (60km/h)   									| 
| Speed limit (60km/h)     			| Speed limit (60km/h) 										|
| Speed limit (100km/h) 1				| Speed limit (100km/h)										|
| Speed limit (100km/h) 2    		| Speed limit (100km/h)					 	  			|
| Stop		                    	| Stop      							|
| No entry			                | No entry      							|
| Slippery road		             	| Slippery road      							|
| Road work			                | Road work      							|
| Children crossing			        | Children crossing      							|
| Wild animals crossing		     	| Wild animals crossing      							|
| Ahead only		               	| Ahead only      							|


The model was able to correctly guess 10 of the 11 traffic signs, which gives an accuracy of 90.9%. The accuracy on the new image set is lower than the given test images.

#### Model Certainty - Softmax Probabilities


The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.  

For image 14.jpg, 17.jpg, 25.jpg, 28.jpg, 3.jpg, 35.jpg, 7.jpg and 7_2.jpg, the probabilities of the correct prediction are all extremely closed to 1 -- the second high probability is below 10^-9. These are all correct predictions.  

For image 31.jpg, the probability of the correct prediction is 99.9%. The second high is around 10^-3.

For image 2.jpg (Speed limit (50km/h)): 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| Speed limit (60km/h)   									| 
| .04     				| Speed limit (80km/h) 										|
| .02					| Speed limit (50km/h)											|
| <10^-3	      			| Speed limit (30km/h)					 				|
| <10^-4				    | Yield     							|

For image 23.jpg (Slippery road): 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .45         			| Slippery road   									| 
| .29     				| Dangerous curve to the right 										|
| .22					| Bicycles crossing											|
| .01	      			| Speed limit (60km/h)					 				|
| .01				    | No passing     							|


---
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### Discussion about the feature maps visual output

'No entry' sign 2nd convolution layer feature maps:  

![c2](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/c2.png)  

'No entry' sign 5th convolution layer feature maps:  

![c5](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/c5.png)  

'No entry' sign maxpooling layer feature maps:  

![mp2](https://github.com/yulongl/p2_TrafficSignClassifier/blob/master/writeup_image/mp2.png)  
