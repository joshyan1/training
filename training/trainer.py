import logging
import argparse
from concurrent import futures
import time
from protos import trainer_pb2_grpc, trainer_pb2
from protos import pokemon_pb2_grpc, pokemon_pb2
import grpc
import threading
import socket
from .pokemon import Pokemon
import utils

class MockTrainer(trainer_pb2_grpc.TrainerServiceServicer):
	def __init__(self, learner_count):
		self.max_learners = learner_count
		self.pokemon = []
		
	def RegisterPokemon(self, request, context):
		print("Called Register Pokemon")
		# Check if the maximum number of Pokémon has been reached
		if len(self.pokemon) >= self.max_learners:
			return trainer_pb2.RegisterPokemonResponse(
				success=False, 
				message="Maximum number of Pokémon registered."
			)
		
		pokemon_stub = pokemon_pb2_grpc.PokemonServiceStub(
                    grpc.insecure_channel(request.network_addr)
                )
		# Add the new Pokémon to the list
		pokemon = Pokemon(len(self.pokemon), request.network_addr, pokemon_stub)
		self.pokemon.append(pokemon)
		logging.info(f"Registered Pokémon {len(self.pokemon)}")

		if len(self.pokemon) == self.max_learners:
			thread = threading.Thread(target=self.start_training)
			thread.start()

		# Return a successful response
		return trainer_pb2.RegisterPokemonResponse(
			success=True, 
			message=f"Pokémon {len(self.pokemon)} registered successfully.",
			id=len(self.pokemon)
		)

	def Training(self, request, context):
		return trainer_pb2.TrainingResultsResponse(
			success = True
		)

	def start_training(self):
		time.sleep(3) # Start up time for the last learner
		logging.info("Starting training across all registered learners.")
		for pokemon in self.pokemon:
			print(f"Calling Pokemon")
			response = pokemon.trainer_stub.StartTraining(pokemon_pb2.StartTrainingRequest())
			print(f"Training returned with success: {response.success} and message: {response.message}")

def serve(learner_count, port=10134):
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	trainer = MockTrainer(learner_count)
	trainer_pb2_grpc.add_TrainerServiceServicer_to_server(trainer, server)
    
	local_ip = utils.get_local_ip()

	#local_ip = socket.gethostbyname(socket.gethostbyname(socket.getfqdn()))
	server_address = f'{local_ip}:{port}'  # Listen on all interfaces
	server.add_insecure_port(server_address)
	server.start()
	#trainer_ngrok_process, trainer_public_url = utils.start_localtunnel(port)
	logging.info(f"Trainer started on {server_address}")
	#logging.info(f"{trainer_public_url}")

	try:
		while True:
			time.sleep(86400)  # Keep the server running
	except KeyboardInterrupt:
		server.stop(0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Initialize your neural network, loss, optimizer, etc.
    learner_count = 2
    serve(learner_count)
			



		
		
	