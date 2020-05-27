#!/usr/bin/env python3


import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, DEVICE, input_size, output_size, num_requests, CPU_EXTENSION=None, plugin=None):

        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Plugin initialization for specified device
        # and load extensions library if specified
        if not plugin:
            log.info("Initializing plugin for {} device...".format(DEVICE))
            self.plugin = IEPlugin(device=DEVICE)

        else:
            self.plugin = plugin

        if CPU_EXTENSION and 'CPU' in DEVICE:
            self.plugin.add_cpu_extension(CPU_EXTENSION)

        # Read IR

        self.net = IENetwork(model=model_xml, weights=model_bin)

        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(self.net)

            unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                print("Unsupported layers found: {}".format(unsupported_layers))
                print("Check whether extensions are available to add to IECore.")
                exit(1)

        if num_requests == 0:
            # Loads network read from IR to the plugin
            self.net_plugin = self.plugin.load(network=self.net)
        else:
            self.net_plugin = self.plugin.load(network=self.net, num_requests=num_requests)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        assert (len(self.net.inputs.keys()) == 1 or len(self.net.inputs.keys()) == 2 or len(
            self.net.inputs.keys()) == 3 or len(self.net.inputs.keys()) == 4 or len(self.net.inputs.keys()) == 5), \
            "Supports only {} input topologies".format(len(self.net.inputs))
        assert len(self.net.outputs) == output_size, \
            "Supports only {} output topologies".format(len(self.net.outputs))

        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):

        self.infer_request_handle = self.net_plugin.start_async(
            request_id=request_id, inputs={self.input_blob: frame})

        return self.net_plugin
        ### Note: You may need to update the function parameters. ###

    def wait(self, request_id):

        wait_process = self.net_plugin.requests[request_id].wait(-1)

        return wait_process
        ### Note: You may need to update the function parameters. ###

    def extract_output(self, request_id, output=None):

        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        return res

        ### Note: You may need to update the function parameters. ###

