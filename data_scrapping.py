from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager  
#import '/home/tanmay/Archana/ENV/archana_env/bin/chromedriver-path' as CHROME_BINARY

year = []
reviews = []
profile_ = []
location=[]
country_name=[]
country = ["Australia",  "Austria",  "Belgium",  "Canada",  "Chile",  "Colombia",  "Czech+Republic",  "Denmark",  "Estonia",  "Finland",  "France",  "Germany",  "Greece",  "Hungary",  "Iceland",  "Ireland",  "Israel",  "Italy",  "Japan",  "Korea",  "Latvia",  "Lithuania",  "Luxembourg",  "Mexico",  "Netherlands",  "New+Zealand",  "Norway",  "Poland",  "Portugal",  "Slovak+Republic",  "Slovenia",  "Spain",  "Sweden",  "Switzerland",  "Turkey",  "United+Kingdom",  "United+States"]

for count in country:

    website = 'https://www.tripadvisor.com/SearchForums?q='+ str(count) +'&s='

    optons = Options()
    optons.add_argument("--headless")
    optons.add_experimental_option("detach", True)

    # optons.binary_location = "/home/tanmay/Archana/paper2"
    # #optons.chromeDriver = "/chromedriver"

    #driver = webdriver.Chrome("/home/tanmay/Archana/paper2/chromedriver", options = optons)
    driver = webdriver.Chrome(service = Service(ChromeDriverManager().install()),
                            options = optons)
    driver.get(website)

    pg_number = 10
    print("Country: ", count)
    print("################### Started ##################")
    while(True):
        try:
            post_links = driver.find_elements(By.XPATH,'//a[@onclick = "setPID(34631)"]')
            ################ Each post
            for i in range(len(post_links)):
                button = driver.find_elements(By.XPATH,'//a[@onclick = "setPID(34631)"]')
                button[i].click()

                reviews_ = driver.find_elements(By.XPATH,'//div[@class ="postBody"]')
                dates = driver.find_elements(By.XPATH,'//div[@class = "postDate"]')
                profile_box = driver.find_elements(By.XPATH, '//div[@class = "profile"]')
                loc = driver.find_elements(By.XPATH, '//div[@class = "location"]')

                ############### Each review
                for j in range(len(reviews_)):
                    
                    ########## Profile
                    try: 
                        p = profile_box[j].text
                        profile = []
                        profile += [p]
                    except:
                        j=j+1
                        continue

                    ########## Review
                    r = reviews_[j].text

                    ########## DATE
                    d = dates[j].text
                    try:
                        diff = int(d.split(' ')[0])
                        y = 2023 - int(diff)
                        
                    except:
                        y = int(d.split(',')[1])

                    ########### Location
                    try:
                        l = loc[j].text
                    except:
                        l = ' '

                    ########## Append    
                    year.append(int(y))
                    reviews.append(r)
                    profile_.append(profile)
                    location.append(l)
                    country_name.append(str(count))

                driver.back()

            ################# Next Page
            next_button = driver.find_element(By.XPATH, '//a[@href = "/SearchForums?q='+ str(count) +'&s= &o='+ str(pg_number)+'"]')
            next_button.click()
            print("On page: ", pg_number)
            pg_number = pg_number+10

        except:
            driver.quit()
            break

    user_id = []
    rating = []

    for i in range(len(profile_)):
        x = profile_[i][0]
        data = x.split('\n')
        if len(data) >1:
            user_id.append(data[0])

            check = data[-1]
            check_2 = check.split(' ')
                    
            if check_2[1] == 'helpful':
                rating.append(check_2[0])
            else:
                w = 'No Rating'
                rating.append(w)

        else:
            user_id.append(' ')
            rating.append(' ')

    print("##################### Finished #######################")

df = pd.DataFrame([])
df['review'] = reviews
df['year'] = year
df['user_id'] = user_id
df['location'] = location
df['rating'] = rating 
df['Country'] = country_name
df.to_csv('Final.csv')
