import motorcortex
import time
import sys


class MCX:
    def __init__(self, hostname="192.168.1.2"):
        self.hostname = hostname
        self.connect()
        self.baseOffsets_old = [228002, 324859, 168898, 145079, -42543]
        self.baseSigns = [1, -1, 1, 1, 1]
        self.offset_params = 5
        self.lenghts_params = 6

    def connect(self):

        parameter_tree = motorcortex.ParameterTree()
        # Open request and subscribe connection
        try:
            self.req, sub = motorcortex.connect("wss://" + self.hostname + ":5568:5567",
                                                motorcortex.MessageTypes(), parameter_tree,
                                                certificate="mcx.cert.pem", timeout_ms=1000,
                                                login="admin", password="vectioneer")
        except RuntimeError as err:
            print(err)
            exit()

    def readOffsets(self):
        offsets = []
        for i in range(self.offset_params):
            get_param_reply_msg = self.req.getParameter(
                'root/Control/actuatorControlLoops/actuatorControlLoop0{0}/drivePositionTransducer/offset'.format(i+1)).get()
            offsets.append(get_param_reply_msg[2][0])
        return offsets

    def readTf(self):

        tfs = []
        for i in range(self.lenghts_params):
            get_param_reply_msg = self.req.getParameter(
                'root/Control/mechanism/Segment0{}/tipPose'.format(i+1)).get()
            tfs.append(get_param_reply_msg[2])

        return tfs

    def setTf(self, newTfs):
        for i in range(self.lenghts_params):
            set_param_reply_msg = self.req.setParameter(
                'root/Control/mechanism/Segment0{}/tipPose'.format(i+1), newTfs[i]).get()
            print(set_param_reply_msg)

    def prepareTfs(self, lenghts):
        tfs = [[]]
        tfs.append([0, 0, lenghts[0]])
        tfs.append([0, 0, lenghts[1]])
        tfs.append([0, 0, lenghts[2]])
        tfs.append([0, lenghts[3], 0])
        tfs.append([0, 0, lenghts[4]])
        tfs.append([0, -lenghts[5], 0])
        return tfs

    def setOffsets(self, newOffsets):

        for i in range(self.offset_params):
            set_param_reply_msg = self.req.setParameter(
                'root/Control/actuatorControlLoops/actuatorControlLoop0{0}/drivePositionTransducer/offset'.format(i+1), int(newOffsets[i])).get()
            print(set_param_reply_msg)

    def calculateNewOffsets(self, angles, baseOffsets):
        newOffsets = []
        for i in range(self.offset_params):
            newOffsets.append(
                float(baseOffsets[i]) + angles[i] * 83443.0268 * self.baseSigns[i])
        return newOffsets


def main():
    mcx = MCX()
    angles = [0.0,
              0.01,
              0.0,
              0.0,
              0.01,
              0.0]
    lenghts = []
    refOffsets = mcx.readOffsets()
    newOffsets = mcx.calculateNewOffsets(angles, refOffsets)
    # mcx.setOffsets(newOffsets)
    len = mcx.readTf()
    a = 0


if __name__ == "__main__":
    main()
