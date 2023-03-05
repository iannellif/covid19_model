import os
import time
from selenium import webdriver
import folium

m = folium.Map(location=[45.5236, -122.6750])

fm = 'index.html'
delay=5
tmpurl='file:///Users/opisthofulax/Dropbox/Flavio-Giulio/SARS-CoV-2/index.html'.format(path=os.getcwd(),mapfile=fm)
m.save(fm)

chdriver = '/Users/opisthofulax/Dropbox/Flavio-Giulio/SARS-CoV-2/chromedriver'
browser = webdriver.Chrome(executable_path=chdriver)
browser.get(tmpurl)
#Give the map tiles some time to load
time.sleep(delay)
browser.save_screenshot('map.png')
browser.quit()


