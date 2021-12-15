# Neural Network

Python is used to develope the neural network along with the help of keras

### To run the code
```
pip install -r requirements.txt
```
```
python3 neuralNet.py
```

Once the code is executed, the files will be loaded from the `optdigits-orig.windep` file which is in the reqpository and the images will be loaded in the format of numbers which will be further used for training.

The data is not converted into images because the model will learn based on the float numbers only. Every 32 lines the input is taken into a input array and the 33rd line is considered to be the target value.

### Model details
```
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_6 (Conv2D)           (None, 30, 30, 100)       1000      
                                                                 
 conv2d_7 (Conv2D)           (None, 28, 28, 75)        67575     
                                                                 
 conv2d_8 (Conv2D)           (None, 26, 26, 50)        33800     
                                                                 
 flatten_3 (Flatten)         (None, 33800)             0         
                                                                 
 dense_3 (Dense)             (None, 10)                338010    
                                                                 
=================================================================
Total params: 440,385
Trainable params: 440,385
Non-trainable params: 0
```