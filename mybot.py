import aiml
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import requests
import json
import wikipediaapi
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import transformer_network
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
import csv
import nltk
import reinforcement_learning
import matplotlib.pyplot as plt
v = """
jordan => ""
irena => ""
ellie => ""
angel => ""
gemma => ""
sarah => ""
jd => ""
british => {}
bulgarian => {}
newzealander => {}
australian => {}
england => England
ireland => Ireland
scotland => Scotland
wales => Wales
bulgaria => Bulgaria
newZealand => NewZealand
newYork => NewYork
australia => Australia
be_in => {}
"""
folval = nltk.Valuation.fromstring(v)
grammar_file = 'simple-sem.fcfg'
objectCounter = 0


def place(person, location, nationality=None):
    global objectCounter

    folval[location]

    if folval[person] == '""':  # clean up if necessary
        o = 'o' + str(objectCounter)
        objectCounter += 1
        folval['o' + o] = o  # insert constant
        folval[person] = o  # insert location information
    else:
        o = folval[person]
        for i in folval["be_in"]:
            if o in i:
                folval["be_in"].remove(i)
                break

    if len(folval["be_in"]) >= 1:  # clean up if necessary
        if ('',) in folval["be_in"]:
            folval["be_in"].clear()

    folval["be_in"].add((o, folval[location]))  # insert location
    if nationality != None:
        if len(folval[nationality]) == 1:  # clean up if necessary
            if ('',) in folval[nationality]:
                folval[nationality].clear()
        folval[nationality].add((o,))

with open('placement.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='"')
    next(reader, None)
    for row in reader:
        person, location, nationality = row[0].split(",")
        place(person, location, nationality)

wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)
weatherAPIKey = "757318177bef8e5078e2543b02adf759"
corpus = ["",
          'How much are flights',
          'How much will flights be',
          'How much will it cost to fly',
          'What is the cost of flying',
          'How much will it cost',
          'How much will the flight cost',
          'What will the flights cost',
          'What will the cost of flying be',
          'Whats the weather like',
          'What is the weather like',
          'Hows the weather today',
          'Hows the weather',
          'How hot is it',
          'How warm is it',
          'How cold is it',
          'Whats it like there'
          'What is it like']
corpusFlights = 8 + 1
corpusWeather = 7 + corpusFlights
corpusWikipedia = 1 + corpusWeather


def updateCSV(person, location):
    with open('placement.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        lines = list(reader)
        newLines = []
        for i in lines:
            i = i[0].split(",")
            if i[0] == person:
                i[1] = location
            i = [",".join(i)]
            newLines.append(i)

    with open('placement.csv', mode='w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
        writer.writerows(newLines)


def predict(filename):
    class_labels = ['airplane', 'automobile', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    model = load_model('trained_model.h5')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    try:
        img = image.load_img(filename, target_size=(32, 32))
    except:
        return None
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    index = model.predict_classes(img)
    return class_labels[index[0]]


def getSimilar(sentence):
    vectorizer = TfidfVectorizer()
    corpus[0] = sentence
    tfidfVector = vectorizer.fit_transform(corpus)
    cosineSimVect = cosine_similarity(tfidfVector[0:1], tfidfVector[1:])
    if np.max(cosineSimVect) > 0.3:
        return np.argmax(cosineSimVect) + 1
    print(transformer_network.predict(sentence))
    return None


def flights(origin, destination, date="2020-03-01", returnDate=None, predict=True):
    try:
        url = "https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/autosuggest/v1.0/UK/GBP/en-GB/"
        headers = {
            'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com",
            'x-rapidapi-key': "7433b6e2c0msh561f7fc11f96c81p1e2a23jsn8bcf4600823a"}
        querystring = {"query": origin}
        response = requests.request(
            "GET", url, headers=headers, params=querystring)
        origin = json.loads(response.text)["Places"][0]["PlaceId"]
        querystring = {"query": destination}
        response = requests.request(
            "GET", url, headers=headers, params=querystring)
        destination = json.loads(response.text)["Places"][0]["PlaceId"]

        print("From: {}\nTo: {}\nOn:{}".format(origin, destination, date))
        if returnDate:
            url = "https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/browsequotes/v1.0/UK/GBP/en-UK/{}/{}/{}/{}".format(
                origin, destination, date, returnDate)
        else:
            url = "https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/browsequotes/v1.0/UK/GBP/en-UK/{}/{}/{}".format(
                origin, destination, date)
        response = json.loads(requests.request(
            "GET", url, headers=headers).text)

        price = response["Quotes"][0]["MinPrice"]
        print("Prices: £{}".format(price))
        return price

    except:
        ## Reinforcement learning section
        
        if predict:
            print("Predicting price, please wait this may take some time.")
            points = []
            prices = [0, 355, 290, 293, 355, 283, 406, 298, 406, 406, 262, 232, 391, 203, 260, 341, 384,
                      232, 335, 233, 249, 257, 231, 195, 263, 245, 409, 249, 255, 234, 201, 266, 269, 226, 192]
            prices = list(enumerate(prices))[1:]
            length = 30
            predict_days = 10
            for i in range(length):
                try:
                    '''
                    skyscanner api is currently down because of covid-19 the line below would work once the api is back on line
                    for now ive got the prices online and manually put them into list prices.
                    '''
                    # points.append([i,flights(origin,destination,date,returnDate,False)])
                    points.append(prices[i])
                except:
                    pass
            print(points)
            hof, func = reinforcement_learning.main(points)
            print("Function to estimate prices:", hof)
            plt.plot(range(1, length), [func(i) for i in range(1, length)])
            plt.plot(range(1, length + predict_days),
                     [func(i) for i in range(1, length + predict_days)])
            plt.ylabel("Amount")
            plt.xlabel("Days Till journey")
            plt.show()


def getWeather(location):
    api_url = "http://api.openweathermap.org/data/2.5/weather?q="
    response = requests.get(api_url + location +
                            "&units=metric&APPID=" + weatherAPIKey)
    try:
        response_json = json.loads(response.content)
        temperature = response_json['main']['temp']
        return temperature
    except:
        return None


def personToLocation(person):
    try:
        g = nltk.Assignment(folval.domain)
        m = nltk.Model(folval.domain, folval)
        e = nltk.Expression.fromstring("be_in(" + person + ",x)")
        sat = m.satisfiers(e, "x", g)
        if len(sat) != 0:
            sol = folval.items()
            for i in sol:
                if i[1] in sat and i[0].isalpha():
                    return (i[0])
        else:
            return person
    except:
        return person

kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot.xml")
masterOrgin = None
masterDestination = None
masterOutbound = None
masterInbound = None
while True:
    try:
        userInput = input("> ")

    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

    answer = kern.respond(userInput)
    if answer == "":
        break
    if answer[0] == "#":
        api = answer[1]
        answer = answer[2:]
        answer = answer.split("$")
        temp = answer.copy()
        answer[0] = personToLocation(answer[0])
        if len(answer) > 1:
            answer[1] = personToLocation(answer[1])
            if answer[0] != temp[0]:
                try:
                    place(temp[0], answer[1])
                    updateCSV(temp[0], answer[1])
                except:
                    print("location currently unsupported")

        answer = "$".join(answer)

        if api == "0":
            location, position = answer.split("$")
            if position == "ORIGIN":
                masterOrgin = location
                print("{} set as orgin".format(masterOrgin))
            elif position == "DESTINATION":
                masterDestination = location
                print("{} set as destination".format(masterDestination))
            elif position == "OUTBOUND":
                try:
                    masterOutbound = location[:10]
                    temp = masterOutbound.split("-")
                    if len(temp[0].strip()) == 4 and len(temp[1].strip()) == 2 and len(temp[2].strip()) == 2:
                        print("{} set as outbound".format(masterOutbound))
                    else:
                        print("Use date format YYYY-MM-DD")
                except:
                    print("Use date format YYYY-MM-DD")
            elif position == "INBOUND":
                try:
                    masterInbound = location[:10]
                    temp = masterInbound.split("-")
                    if len(temp[0].strip()) == 4 and len(temp[1].strip()) == 2 and len(temp[2].strip()) == 2:
                        print("{} set as inbound".format(masterInbound))
                    else:
                        print("Use date format YYYY-MM-DD")
                except:
                    print("Use date format YYYY-MM-DD")
            else:
                print("Unable to set {}".format(position))

        elif api == "1":
            if len(answer.split("$")) == 2:
                masterOrgin, masterDestination = answer.split("$")
                flights(masterOrgin, masterDestination)
            if len(answer.split("$")) == 3:
                masterOrgin, masterDestination, masterOutbound = answer.split(
                    "$")
                flights(masterOrgin, masterDestination, masterOutbound)

        elif api == "2":
            weather = getWeather(answer)
            if weather:
                print("The tempreture in {} is {}".format(answer, weather))
        elif api == "3":
            wpage = wiki.page(answer)
            if wpage.exists():
                print(wpage.summary)
            else:
                print("Sorry, I don't know what that is.")

        elif api == "4":
            answer = answer.replace("  #9$", ".")
            if "jpg" not in answer and not "png" in answer:
                print("Invalid File type")
            else:
                prediction = predict(answer)
                if prediction == "automobile":
                    print(
                        "This is a Automobile, why not travel to Germany to drive on the world famous Autobahn")
                    if masterOrgin:
                        flights(masterOrgin, "germany")

                if prediction == "airplane":
                    print(
                        "This is a Airplane, why not ask me about flight infomation to book your next holiday")

                if prediction == "bird":
                    print(
                        "This is a Bird, why not travel The Jurong Bird Park in Singapore to visit the worlds largest bird sancturay")
                    if masterOrgin:
                        flights(masterOrgin, "singapore")

                if prediction == "cat":
                    print(
                        "This is a Cat, why not travel to Tashirojima to see the cat island")
                    if masterOrgin:
                        flights(masterOrgin, "japan")

                if prediction == "deer":
                    print(
                        "This is a Deer, why not travel to canada to see herds of deer freely roaming")
                    if masterOrgin:
                        flights(masterOrgin, "canada")

                if prediction == "dog":
                    print(
                        "This is a Dog, why not visit dog island, just off the coast of florida")
                    if masterOrgin:
                        flights(masterOrgin, "florida")

                if prediction == "frog":
                    print(
                        "This is a Frog, why not travel the amazon rain forest home to thousand of species of frogs")
                    if masterOrgin:
                        flights(masterOrgin, "Rio de Janeiro")

                if prediction == "horse":
                    print(
                        "This is a horse, why not travel to Iceland to see the wild Icelandic horses")
                    if masterOrgin:
                        flights(masterOrgin, "iceland")

                elif prediction == None:
                    print("Could not find file {}".format(answer))
                else:
                    wpage = wiki.page(answer)
                    if wpage.exists():
                        print(wpage.summary)

        elif api == "5":  # Who is in  ...
            try:
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                e = nltk.Expression.fromstring("be_in(x," + answer + ")")
                sat = m.satisfiers(e, "x", g)
                if len(sat) == 0:
                    print("None.")
                else:
                    sol = folval.items()
                    for i in sol:
                        if i[1] in sat and i[0].isalpha():
                            print(i[0])
            except:
                print("Error please try again")

        elif api == "6":  # Are there any x in y
            try:
                params = answer.split("$")
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'some ' + params[1] + ' are_in ' + params[2].title()
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2] == True:
                    print("Yes.")
                else:
                    print("No.")
            except:
                print("Error please try again")

        elif api == "7":  # Are all x in y
            try:
                params = answer.split("$")
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'all ' + params[1] + ' are_in ' + params[2].title()
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2] == True:
                    print("Yes.")
                else:
                    print("No.")
            except:
                print("Error please try again")

        elif api == "9":
            correctedAnswer = getSimilar(answer)
            if correctedAnswer:
                if correctedAnswer < corpusFlights:
                    if masterOrgin and masterDestination:
                        flights(masterOrgin, masterDestination,
                                masterOutbound, masterInbound)
                    elif masterOrgin:
                        print("Please set an destination")
                    elif masterDestination:
                        print("Please set an origin")
                    else:
                        print("Please set an origin and destination")
                elif correctedAnswer < corpusWeather:
                    if masterOrgin:
                        weatherOrigin = getWeather(masterOrgin)
                    else:
                        weatherOrigin = None
                    if masterDestination:
                        weatherDestination = getWeather(masterDestination)
                    else:
                        weatherDestination = None
                    if weatherOrigin:
                        print("The tempreture in {} is {}".format(
                            masterOrgin, weatherOrigin))
                    if weatherDestination:
                        print("The tempreture at {} is {}".format(
                            masterDestination, weatherDestination))
                    if not (weatherOrigin or weatherDestination):
                        print("Could not get weather infomation")
                elif correctedAnswer < corpusWikipedia:
                    if masterDestination:
                        wpage = wiki.page(masterDestination)
                        if wpage.exists():
                            print(wpage.summary)
                            print("Learn more at", wpage.canonicalurl)
                        else:
                            print("Sorry, I don't know what {} is.".format(
                                masterDestination))
                    else:
                        print("Please set an destination")
    else:
        print(answer)
