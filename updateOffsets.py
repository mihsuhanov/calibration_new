import motorcortex
import time
import sys
import json


class MCX:
    def __init__(self, hostname="192.168.56.101", filename="results.json"):
        self.hostname = hostname
        self.connect()
        # self.baseOffsets_old = [0, 0, -131072, 0, -131072, 0]
        self.baseSigns = [1, -1, 1, -1, 1, 1]
        self.motors = 6

        with open(filename, "r") as file:
            params = json.load(file)
            self.mcx_params = params["mcx_params"]
            self.mcx_params[0][3] = 0
            self.offsets = params["offsets"]
            self.tool_params = params["estimated_tool_params"]

    def connect(self):

        parameter_tree = motorcortex.ParameterTree()
        # Open request and subscribe connection
        try:
            self.req, sub = motorcortex.connect("wss://" + self.hostname + ":5568:5567",
                                                motorcortex.MessageTypes(), parameter_tree,
                                                certificate="mcx.cert.crt", timeout_ms=1000,
                                                login="admin", password="vectioneer")
        except RuntimeError as err:
            print(err)
            exit()

    def readOffsets(self):
        offsets = []
        for i in range(self.motors):
            get_param_reply_msg = self.req.getParameter(
                f'root/AxesControl/actuatorControlLoops/actuatorControlLoop0{i+1}/positionTransformation/transducer/offset').get()
            offsets.append(get_param_reply_msg[2][0])
        return offsets

    def setAll(self, newOffsets):
        for i in range(self.motors):
            self.req.setParameter(f'root/AxesControl/actuatorControlLoops/actuatorControlLoop0{i+1}/positionTransformation/transducer/offset', int(newOffsets[i])).get()
            self.req.setParameter(f'root/ManipulatorControl/mechanism/segment{i+1}/tippose', self.mcx_params[i]).get()
        self.req.setParameter(f'root/ManipulatorControl/mechanism/tool/tippose', self.tool_params).get()
                
    def calculateNewOffsets(self, baseOffsets):
        newOffsets = []
        for i in range(self.motors):
            newOffsets.append(
                float(baseOffsets[i]) + self.offsets[i] * 83443.0268 * self.baseSigns[i])
        return newOffsets


def main():
    mcx = MCX()
    refOffsets = mcx.readOffsets()
    newOffsets = mcx.calculateNewOffsets(refOffsets)
    mcx.setAll(newOffsets)


if __name__ == "__main__":
    main()
