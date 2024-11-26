# Arceus

Hi everyone! Welcome to our distributed training framework, built from first principles leveraging model parallelism to optimize training on Apple M-series clusters! Right now, please train our FFN with the MNIST dataset!

## Run
To run this version of Arceus, please first initialize a virtual environment and install all required modules
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

First initialize the Flask server to handle device registration and training orchestration
```shell
python -m api.server
```

Now, register your devices. There must be atleast 1 and at most the number equal to the layers of your model.
On each of your devices, run
```shell
python -m nn.device_client
```

Then, initialize your devices with the layers of the model
```shell
curl -X POST http://localhost:4000/api/network/initialize
```

Finally, train with the following request
```shell
curl -X POST http://localhost:5000/api/network/train \
     -H "Content-Type: application/json" \
     -d '{"epochs": 10, "learning_rate": 0.1}'
```

You will see a message on your server
```
127.0.0.1 - - [26/Nov/2024 10:40:56] "POST /api/network/train HTTP/1.1" 200 -

Starting distributed training across devices...
Learning rate: 0.1
Quantization bits: 8
Device: mps
```

## Network Setup

To run Arceus across multiple machines on your local network:

1. Start the API server on your main machine:
```shell
python -m api.server
```