import bluetooth

# install pyblues

class BluetoothComm(object):
    
    def __init__(self, serverMACAddress):
        self.serverMACAddress = serverMACAddress
        self.port = 3
        print("Starting connection")
        self.s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.s.connect((serverMACAddress, self.port))
        print("connected")
        
    def send(self, msg):
        self.s.send(msg)
        return True
        
    def close(self):
        self.s.close()
        return True


# FOR TESTING
if(__name__ == '__main__'):
    obj = BluetoothComm('E4:46:DA:B1:84:AC')
    obj.send("asd")
    obj.close()