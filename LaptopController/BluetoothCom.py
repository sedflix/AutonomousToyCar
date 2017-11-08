import bluetooth

# install pyblues

class BluetoothComm(object):
    
    def __init__(self, serverMACAddress, debug):
        self.serverMACAddress = serverMACAddress
        self.port = 1
        self.debug = debug
        print("Starting connection")
        self.s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.s.connect((serverMACAddress, self.port))
        print("connected")
        
    def send(self, msg):
        self.s.send(msg)
        if self.debug:
            print (msg)
            pass
        return True
        
    def close(self):
        self.s.close()
        return True

# FOR TESTING
if(__name__ == '__main__'):
    obj = BluetoothComm('00:15:83:35:99:09', True)
    obj.send("a180")
    obj.close()