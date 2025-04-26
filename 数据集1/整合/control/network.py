from p4utils.mininetlib.network_API import NetworkAPI

net = NetworkAPI()

# Network general options
net.setLogLevel('info')
net.setCompiler(p4rt=True)
net.disableArpTables()

# Network definition
net.addP4RuntimeSwitch('s1')
net.setP4Source('s1','s1.p4')
net.addHost('h1')
net.addHost('h2')
net.addLink('h1','s1')
net.addLink('h2','s1')

# Assignment strategy
net.l2()

# Nodes general options
net.enablePcapDumpAll()
net.enableLogAll()
net.enableCli()
net.startNetwork()