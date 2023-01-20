# sudo pip3 install adafruit-circuitpython-dht
import adafruit_dht
import time

dht_device = adafruit_dht.DHT22(4) # 4는 pin번호

while(True):
    try:
            temperature = dht_device.temperature # 온도값 가져오기
            humidity = dht_device.humidity # 습도값 가져오기
            print("Temp: " + str(temperature) + "C, Humidity: " + str(humidity) + "%")
            time.sleep(3)

    except RuntimeError as error:

        print(error.args[0])

    except Exception as error:
        dht_device.exit()
        raise error
