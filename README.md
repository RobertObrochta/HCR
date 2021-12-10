# Handwriting Character Recognition
## Model Configuration

### To-Do
- [ ] Test on STM32MPU - real-time inferencing
- [x] Figure out original model from STMCubeMX-AI Analysis
- [x] Test to see if this model predicts with consistency
- [x] Visualization - accuracy confusion matrix for predictions (heatmap)
- [X] Visualization for MNIST dataset
- [X] Added quantization within TFLite


#### Notes
Quantization was achieved from *float32* to *uint8* datatypes via TensorFlow Lite. This quantized model is outputted as a **.tflite** file within the **./results/** directory. Additionally, quantization may be achieved within STM32CubeMX from the **.h5** file, which has *unquantized, float32* weights. 


### Prediction Confusion Matrix Output
![Sample Output Matrix](/images/samplematrix.png)


### Citations
**EMNIST** - Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
