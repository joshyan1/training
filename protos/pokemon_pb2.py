# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: pokemon.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'pokemon.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rpokemon.proto\x12\x07pokemon\"1\n\x11InitDeviceRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x0e\n\x06layers\x18\x02 \x01(\x05\"6\n\x12InitDeviceResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"\x16\n\x14StartTrainingRequest\"9\n\x15StartTrainingResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\x32\xa7\x01\n\x0ePokemonService\x12\x45\n\nInitDevice\x12\x1a.pokemon.InitDeviceRequest\x1a\x1b.pokemon.InitDeviceResponse\x12N\n\rStartTraining\x12\x1d.pokemon.StartTrainingRequest\x1a\x1e.pokemon.StartTrainingResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'pokemon_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_INITDEVICEREQUEST']._serialized_start=26
  _globals['_INITDEVICEREQUEST']._serialized_end=75
  _globals['_INITDEVICERESPONSE']._serialized_start=77
  _globals['_INITDEVICERESPONSE']._serialized_end=131
  _globals['_STARTTRAININGREQUEST']._serialized_start=133
  _globals['_STARTTRAININGREQUEST']._serialized_end=155
  _globals['_STARTTRAININGRESPONSE']._serialized_start=157
  _globals['_STARTTRAININGRESPONSE']._serialized_end=214
  _globals['_POKEMONSERVICE']._serialized_start=217
  _globals['_POKEMONSERVICE']._serialized_end=384
# @@protoc_insertion_point(module_scope)
