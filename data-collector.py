from bs4 import BeautifulSoup
import urllib.request as request
import cv2
from selenium import webdriver
import os

CATEGORIES = ["Donald Trump","Narender Modi","Obama","George Bush"]
NUM_OF_IMAGES = 20

driver = webdriver.Chrome("drivers/chromedriver.exe")
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

try:
	os.mkdir("data")
except Exception as e:
	pass

os.chdir("data")

for category in CATEGORIES:
	try:
		os.mkdir(category)
	except Exception as e:
		pass
	count = 1
	base_url="https://www.google.com/search?biw=1366&bih=625&tbm=isch&sa=1&ei=b8yNXPrADcnfz7sPnPWsYA&q="+category.replace(" ","+")+"&oq=sachin&gs_l=img.1.0.0i67l2j0l5j0i67j0j0i67.18175568.18176912..18178621...0.0..0.160.796.0j6......1....1..gws-wiz-img.......35i39.DBMPRGPvO3c"
	driver.get(base_url)
	driver.execute_script("window.scrollTo(0, 10000000)") 
	html = driver.page_source
	soup = BeautifulSoup(html,"html.parser")
	images = soup.findAll("img")
	print(len(images))
	for img in images:
		link = img.get("data-src")
		if link==None:
			link=img.get("src")
		try:
			if link.startswith('/images'):
				link = "https://www.google.com"+link

			image = request.urlopen(link).read()
			image_file = open(category+"/"+str(count)+".jpg","wb")
			image_file.write(image)
			image = cv2.imread(category+"/"+str(count)+".jpg")

			faces = face_cascade.detectMultiScale(image, 1.5, 5)
			
			if len(faces)==1:
				count = count + 1
			if count > NUM_OF_IMAGES:
				break
		except Exception as e:
			print(e)
			pass
