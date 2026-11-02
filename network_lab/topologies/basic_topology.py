from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.node import OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel


class StreamTopology(Topo):

    def build(self):

        # Create server and client
        server = self.addHost(
            "h1",
            ip="10.0.0.1/24"
        )

        client = self.addHost(
            "h2",
            ip="10.0.0.2/24"
        )

        # Create virtual switch
        switch = self.addSwitch(
            "s1",
            failMode="standalone"
        )

        # Server-to-switch link
        self.addLink(
            server,
            switch,
            cls=TCLink,
            bw=2,
            delay="5ms"
        )

        # Switch-to-client link
        self.addLink(
            switch,
            client,
            cls=TCLink,
            bw=2,
            delay="5ms"
        )


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

    print("\nStreamIQ Virtual Network Started!")
    print("Server: 10.0.0.1")
    print("Client: 10.0.0.2")

    CLI(net)

    net.stop()


if __name__ == "__main__":

    setLogLevel("info")
    run()