# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: neural_network.proto
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
    'neural_network.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14neural_network.proto\x12\x0eneural_network\"J\n\x0bLayerConfig\x12\x12\n\ninput_size\x18\x01 \x01(\x05\x12\x13\n\x0boutput_size\x18\x02 \x01(\x05\x12\x12\n\nactivation\x18\x03 \x01(\t\"Z\n\x11InitializeRequest\x12\x32\n\rlayer_configs\x18\x01 \x03(\x0b\x32\x1b.neural_network.LayerConfig\x12\x11\n\tdevice_id\x18\x02 \x01(\x05\"5\n\x12InitializeResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\x0f\n\x07message\x18\x02 \x01(\t\"G\n\x0e\x46orwardRequest\x12\r\n\x05input\x18\x01 \x03(\x02\x12\x12\n\nbatch_size\x18\x02 \x01(\x05\x12\x12\n\ninput_size\x18\x03 \x01(\x05\"J\n\x0f\x46orwardResponse\x12\x0e\n\x06output\x18\x01 \x03(\x02\x12\x12\n\nbatch_size\x18\x02 \x01(\x05\x12\x13\n\x0boutput_size\x18\x03 \x01(\x05\"M\n\x0f\x42\x61\x63kwardRequest\x12\x12\n\ngrad_input\x18\x01 \x03(\x02\x12\x12\n\nbatch_size\x18\x02 \x01(\x05\x12\x12\n\ninput_size\x18\x03 \x01(\x05\"P\n\x10\x42\x61\x63kwardResponse\x12\x13\n\x0bgrad_output\x18\x01 \x03(\x02\x12\x12\n\nbatch_size\x18\x02 \x01(\x05\x12\x13\n\x0boutput_size\x18\x03 \x01(\x05\"&\n\rUpdateRequest\x12\x15\n\rlearning_rate\x18\x01 \x01(\x02\" \n\x0eUpdateResponse\x12\x0e\n\x06status\x18\x01 \x01(\t2\xd7\x02\n\x14NeuralNetworkService\x12U\n\nInitialize\x12!.neural_network.InitializeRequest\x1a\".neural_network.InitializeResponse\"\x00\x12L\n\x07\x46orward\x12\x1e.neural_network.ForwardRequest\x1a\x1f.neural_network.ForwardResponse\"\x00\x12O\n\x08\x42\x61\x63kward\x12\x1f.neural_network.BackwardRequest\x1a .neural_network.BackwardResponse\"\x00\x12I\n\x06Update\x12\x1d.neural_network.UpdateRequest\x1a\x1e.neural_network.UpdateResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'neural_network_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_LAYERCONFIG']._serialized_start=40
  _globals['_LAYERCONFIG']._serialized_end=114
  _globals['_INITIALIZEREQUEST']._serialized_start=116
  _globals['_INITIALIZEREQUEST']._serialized_end=206
  _globals['_INITIALIZERESPONSE']._serialized_start=208
  _globals['_INITIALIZERESPONSE']._serialized_end=261
  _globals['_FORWARDREQUEST']._serialized_start=263
  _globals['_FORWARDREQUEST']._serialized_end=334
  _globals['_FORWARDRESPONSE']._serialized_start=336
  _globals['_FORWARDRESPONSE']._serialized_end=410
  _globals['_BACKWARDREQUEST']._serialized_start=412
  _globals['_BACKWARDREQUEST']._serialized_end=489
  _globals['_BACKWARDRESPONSE']._serialized_start=491
  _globals['_BACKWARDRESPONSE']._serialized_end=571
  _globals['_UPDATEREQUEST']._serialized_start=573
  _globals['_UPDATEREQUEST']._serialized_end=611
  _globals['_UPDATERESPONSE']._serialized_start=613
  _globals['_UPDATERESPONSE']._serialized_end=645
  _globals['_NEURALNETWORKSERVICE']._serialized_start=648
  _globals['_NEURALNETWORKSERVICE']._serialized_end=991
# @@protoc_insertion_point(module_scope)