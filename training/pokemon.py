import logging
import argparse
from concurrent import futures
import time
from protos import trainer_pb2_grpc, trainer_pb2
from protos import pokemon_pb2_grpc, pokemon_pb2
import grpc
import socket
import utils

class Pokemon(pokemon_pb2_grpc.PokemonServiceServicer):
	def __init__(self, device, network_addr, trainer_stub):
		self.device = device
		self.network = network_addr
		self.trainer_stub = trainer_stub

	def InitDevice(self, request, context):
		self.device.data = request.data
		return pokemon_pb2.InitDeviceResponse(
			success = True,
			message = "Device initialized"
		)

	def StartTraining(self, request, context):
		comp = self.device.forward()
		return pokemon_pb2.StartTrainingResponse (
			message = comp,
			success = True
		)

class Device:
	def __init__(self, layers, id):
		self.data = []
		self.layers = layers
		self.id = id

	def forward(self):
		print(f"{self.id} is forwarding")   
		return(f"{self.id} is forwarding")

def serve(network_addr, port, leader_stub, id):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    device = Device(5, id)
    pokemon = Pokemon(device, network_addr, leader_stub)

    print("Tring to add service")
    pokemon_pb2_grpc.add_PokemonServiceServicer_to_server(pokemon, server)
    logging.info(f'Learner started on {network_addr}')
    print(f'Learner started on {network_addr}')
    print(f"{port}")
    server.add_insecure_port(f'0.0.0.0:{port}')

    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Pokemon Service')
    parser.add_argument('--leader-address', type=str, help='Network address of the leader in the form 192.168.xxx.xxx:xxxx')
    parser.add_argument('--port', type=int, default=10135, help='Port you want the learner to use')
    args = parser.parse_args()

    trainer_address = "10.36.159.40:10134"
    #grpc_target = trainer_address.replace("https://", "")
    pokemon_port = args.port
    
    print("finding channel")
    channel = grpc.insecure_channel(f"{trainer_address}")  # Add the port manually
    print("channel found")
    trainer_stub = trainer_pb2_grpc.TrainerServiceStub(channel)
    print("trainer_stub added")
    print(pokemon_port)
    #local_ip = utils.start_localtunnel(pokemon_port)
    local_ip = utils.get_local_ip()

    network_addr = f"{local_ip}:{pokemon_port}"
    print(local_ip)

    print("Registering learner")
    logging.info('Registering learner...')
    is_registered = trainer_stub.RegisterPokemon(
            trainer_pb2.RegisterPokemonRequest(
                network_addr=network_addr,
                id=1
            )
    )

    if is_registered.success:
        logging.info(f"{is_registered.message}")
        serve(network_addr, pokemon_port, trainer_stub, is_registered.id)
    else:
        logging.error('Registering learner unsuccessful')