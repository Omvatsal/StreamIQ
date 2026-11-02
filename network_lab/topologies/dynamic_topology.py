
import random
import time

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.node import OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel


class StreamTopology(Topo):

    def build(self):

        server = self.addHost(
            "h1",
            ip="10.0.0.1/24"
        )

        client = self.addHost(
            "h2",
            ip="10.0.0.2/24"
        )

        switch = self.addSwitch(
            "s1",
            failMode="standalone"
        )

        self.addLink(
            server,
            switch,
            cls=TCLink,
            bw=10,
            delay="5ms"
        )

        self.addLink(
            switch,
            client,
            cls=TCLink,
            bw=10,
            delay="5ms"
        )


class StreamCLI(CLI):

    def __init__(self, net):

        self.net = net
        self.profile_running = False

        super().__init__(net)

    def set_bandwidth(self, bandwidth):

        # Video flows from h1 (server) to h2 (client).
        # Therefore, control server-side egress.

        server_intf = self.net.get("h1").intf("h1-eth0")

        server_intf.config(
            bw=bandwidth,
            delay="5ms"
        )

        print(
            f"\n[StreamIQ] Server egress bandwidth: "
            f"{bandwidth:.2f} Mbps"
        )

    def do_setbw(self, line):
        """
        Manually change bandwidth.

        Usage: setbw 3
        """

        try:
            bandwidth = float(line.strip())

            if not 0.1 <= bandwidth <= 10:
                raise ValueError(
                    "Bandwidth must be between 0.1 and 10 Mbps"
                )

            self.set_bandwidth(bandwidth)

        except ValueError as error:
            print(f"Invalid bandwidth: {error}")

    def do_profile(self, line):
        """
        Run a randomized network profile.

        Usage: profile
        """

        if self.profile_running:
            print("A profile is already running.")
            return

        self.profile_running = True

        try:

            # Configuration
            min_bandwidth = 1.0
            max_bandwidth = 10.0

            min_duration = 4
            max_duration = 9

            phases = 8

            print("\n[StreamIQ] Random profile started\n")

            for phase in range(1, phases + 1):

                bandwidth = round(
                    random.uniform(
                        min_bandwidth,
                        max_bandwidth
                    ),
                    2
                )

                duration = random.randint(
                    min_duration,
                    max_duration
                )

                self.set_bandwidth(bandwidth)

                print(
                    f"[Phase {phase}/{phases}] "
                    f"Bandwidth: {bandwidth} Mbps | "
                    f"Duration: {duration}s"
                )

                time.sleep(duration)

            print("\n[StreamIQ] Random profile completed!")

        finally:
            self.profile_running = False


def run():

    topo = StreamTopology()

    net = Mininet(
        topo=topo,
        link=TCLink,
        switch=OVSSwitch,
        controller=None,
        autoSetMacs=True
    )

    net.start()

    print("\n===================================")
    print("       StreamIQ Dynamic Network")
    print("===================================")

    print("Server: 10.0.0.1")
    print("Client: 10.0.0.2")

    print("\nCommands:")
    print("  setbw 2  - Manually set bandwidth")
    print("  profile  - Start randomized profile")

    print("===================================\n")

    try:
        StreamCLI(net)

    finally:
        net.stop()


if __name__ == "__main__":

    setLogLevel("info")
    run()
