# Image_classification

This repo contains the python code to train and run an image classification network.
Test data labels, and test data softmax are available in the root folder as .pt files.


## usage

### argparse
Usage: python main.py [-h] [-t] [--model_path MODEL_PATH] [-td] [--task2] [--task3]

```
optional arguments:
  -h, --help            show this help message and exit
  -t, --train           Train the model (default: False)
  --model_path MODEL_PATH
                        Path to save the trained model (default: None)
  -td, --test_data      Use a small dataset for testing (default: False)
  --task2               Run task 2 (default: False)
  --task3               Run task 3 (default: False)
```

## running 
Install required packages using 

```python
pip3 install -r requirements.txt
```

To run this program, run main.py.

If [-t, --train] is not active, the evaluation parth will start using a pretrained model that is found in the models directory.
This requires no path setup as a path to a pretrained model is already coded for evaluation.

If you want to train your own model, you can add the [-t, --train] argument. IMPORTAINT, everything is seeded
This also requires [--model_path] to be added, and is the path the newly trained model will be saved to.
If you want to evaluate this new model, you need to change the path in main.py

