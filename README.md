# OCR_and_STR
This project has been finished successfully, and some issues in work are listed as follow:

1. the handling in data
2. using optimizer Adam, the default learning rate in official code is 0.01, while this learning rate will result in the result is always '-------------------------------' no matter with the number of epoch or data. so the correct learning rate is 0.0001. But I don't find the reason.
3. the final result is :
### VGG+LSTM
Test loss: 0.000872, accuray: 0.811200
### Resnet+LSTM
Test loss: 0.007662, accuray: 0.827853



