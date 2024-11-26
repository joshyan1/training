# API

## Run
To start you Flask Server, run
```shell
python -m api.server
```

## Endpoints
- [Register Device](#register-device)
- [Get Devices](#get-devices)
- [Initialize Network](#initialize-neural-network)
- [Train Network](#train-neural-network)
- [Unregister Device](#unregister-device)


### Register Device
Registers a new device with the server
```
POST /api/devices/register
```

### Get Devices
Retrieves all registered devices 
```
GET /api/devices
```

### Initialize Neural Network
Initializes neural network by allocating layers to registered devices
```
POST /api/network/initialize
```


### Train Neural Network
Begin training of neural network
```
POST /api/network/train
```

### Unregister Device
Remove a registered device from the server
```
DELETE /api/devices/<int:port>
```
