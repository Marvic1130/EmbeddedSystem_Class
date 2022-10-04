# sudo pip3 install adafruit-circuitpython-dht
import adafruit_dht
import time

dht_device = adafruit_dht.DHT11(4) # 4는 pin번호


while(True):
    if time.time()%5 == 0: # 5초에 한번
        temperature = dht_device.temperature # 온도값 가져오기
        humidity = dht_device.humidity # 습도값 가져오기
        print("Temp: " + str(temperature) + "C, Humidity: " + str(humidity) + "%")