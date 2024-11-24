# Arceus

Hi everyone! Welcome to our distributed training framework, built from first principles leveraging model parallelism to optimize training on Apple M-series clusters! Right now, please train our MNIST neural network!

## Run
To run this version of Arceus, please first initialize a virtual environment and install all required modules
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Now, to start training, initialize 3 separate terminals and run the following commands on each
```
cd nn
python3 device_server.py [port]
```

Finally, run the `coordinator.py` file
```
python3 coordinator.py
```