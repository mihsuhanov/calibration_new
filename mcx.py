import motorcortex
import time
import sys
import csv
# Define the callback function that will be called whenever a message is received


def message_received(parameters):
    # for cnt in range(0, len(parameters)):
    #     param = parameters[cnt]
    #     timestamp = param.timestamp.sec + param.timestamp.nsec * 1e-9
    #     value = param.value
    # print the timestamp and value; convert the value to a string first
    # so we do not need to check all types before printing it
    # print(f"Notify: {timestamp}, {value}")
    return


def main():

    header = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6',
              'px', 'py', 'pz', 'fk_x', 'fk_y', 'fk_z']
    f = open('orientation.csv', 'w', encoding='UTF8')
    writer = csv.writer(f)
    writer.writerow(header)

    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()

    parameter_tree = motorcortex.ParameterTree()
    # Open request and subscribe connection
    try:
        req, sub = motorcortex.connect("wss://192.168.1.2:5568:5567",
                                       motorcortex.MessageTypes(), parameter_tree,
                                       certificate="mcx.cert.pem", timeout_ms=1000,
                                       login="admin", password="vectioneer")
    except RuntimeError as err:
        print(err)
        exit()

    paths = ['root/Control/fkActualToolCoord/jointPositions',
             'root/Control/fkActualToolCoord/toolCoordinates']

    divider = 100

    subscription = sub.subscribe(paths, 'group1', divider)
    is_subscribed = subscription.get()
    # print subscription status and layout
    if (is_subscribed is not None) and (is_subscribed.status == motorcortex.OK):
        print(f"Subscription successful, layout: {subscription.layout()}")
    else:
        print(f"Subscription failed, do your paths exist? \npaths: {paths}")
        sub.close()
        exit()

    # set the callback function that handles the received data
    # Note that this is a non-blocking call, starting a new thread that handles
    # the messages. You should keep the application alive for a s long as you need to
    # receive the messages
    # subscription.notify(message_received)

    # polling subscription
    try:
        for i in range(5000):
            value = subscription.read()
            # if value:
            #     print(
            #         f"Polling, timestamp: {value[0].timestamp} value: {value[0].value}")
            angles = v.devices["tracker_1"].get_pose_euler()
            data = list(value[0][1])
            data_dp = list(value[1][1])
            if data_dp[0] == 0:
                continue
            data.extend([angles[0], angles[2], angles[1]])
            data.extend(data_dp[:3])
            writer.writerow(data)
            time.sleep(0.05)
    except KeyboardInterrupt:
        sub.close()
        f.close()
    sub.close()
    f.close()

    # close the subscription when done


if __name__ == "__main__":
    main()
