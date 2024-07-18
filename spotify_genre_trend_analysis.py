#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: alexandra
"""
import requests # for POST and GET requests for API
import pandas as pd
import numpy as np
import base64
import seaborn as sns # For visualization tools
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller # For ADF test
from sklearn.preprocessing import MinMaxScaler # For scaling data (LSTM)
from sklearn.model_selection import train_test_split # For splitting, training, and testing data (LSTM)
from tensorflow.keras.models import Sequential # For LSTM
from tensorflow.keras.layers import LSTM, Dense # for LSTM


# redirect URL: http://localhost:3000

def GetToken(redirect_uri, client_id, client_secret, scope):
    token_ep_url = "https://accounts.spotify.com/api/token" # Token endpoint url: where we are sending the POST request to
    
    authorization_string = client_id + ":" + client_secret
    authorization_bytes = authorization_string.encode("utf-8")
    authorization_base64 = str(base64.b64encode(authorization_bytes), "utf-8")
    
    content_header = {
        "Authorization" : "Basic " + authorization_base64,
        "Content-Type" : "application/x-www-form-urlencoded"
        }

    data = {
            "grant_type" : "client_credentials",
            "client_id" : f"{client_id}",
            "client_secret" : f"{client_secret}",
            "scope" : scope
            }
    
    request = requests.post(token_ep_url, headers = content_header, data = data)
    
    # Checking if the POST request was successful by its request code (has to be 200)
    if request.status_code == 200: # Its successful, so lets get the access token
        print("Getting access token (valid for an hour)")
        
        json_response = request.json()
        #print(json_response, "\n")
        token = json_response.get("access_token")
        #refresh_token = json_response.get("refresh token")
        #print("\n")
        #print(f"{access_token}")
        #print("\nRefresh token:")
        #print(f"{refresh_token}")
        #json_response = json.loads(request.content)
        #token = json_response["access_token"]
        return token # good
    elif request.status_code == 400: # accidentally put 400 but its true
        print("no")
        return
    else:
        print("bad")
        return
        
def GetRefreshToken(client_id, client_secret): # Gets a refresh token for future authorization once the original token expires
    authorization_url = f"https://accounts.spotify.com/authorize?client_id={client_id}&response_type=code&redirect_uri={redirect_uri}&scope=user-read-private%20user-read-email%20offline_access"

def GetHeader(access_token): # To get the standard header for authorization use (Auth: Bearer (access_token)) # good
        header = {"Authorization" : f"Bearer {access_token}"}
        return header;

def POSTRequest(url, header): # Gets POST request from given URL and tests if it was successful or not; for redundancy
    request = requests.get(url, headers = header) # Getting data
    
    # Checking if request was successful
    if request.status_code == 200:
        json_response = request.json()
        return json_response # returns a dictionary
    else:
        return f"Unsuccessful request, error code: {request.status_code}" # find out how to return only the entry not both

def SearchGenreTracks(genre, limit, header, market, offset, year_range = None): # Mainly for use with SearchByGenre # Good
    url = f'https://api.spotify.com/v1/search?q=genre:{genre}'
    
    # Checking if we have a year given (i.e., for GenreTrendAnalysis)
    if year_range: # Meaning we have a year range given
        url += f' year:{year_range}' # good
    url += f'&type=track&market={market}&limit={limit}&offset={offset}'
    
    request = POSTRequest(url, header) # Gets search
    #print(type(request))
    return request

def ExtractTrackID(track_name, artist_name = None): # In use with the SearchItems function to retrieve the track ID by track name
        # 1. Get the result from SearchItems function
        # 2. Filter the result to organize into a dictionary the artist name, name of the track matching the given one, and track ID
            # -> note that this will probably have to be done one at a time through a for loop other functions, storing the....
            # ... results in a dictionary for safe keeping
        # 3. return the ID (then from here for this purpose, go to the function to retrieve the audio features)
        print("\nIn ExtractTrackID")
        print(f"artist_name in ExtractTrackID: {artist_name}") # good
        # Program
        search = SearchItems(access_token, track_name, "track", artist_name) # good so far
        print("\nBack in ExtractTrackID")
        #print("\n\n\nSearch result:\n", search) # good gives us the search result
        #print(search['tracks']['items'][0]) # 0 is max key because my limit is always 1; can go higher if limit is diff
        search_item = search['tracks']['items'][0] # to access first and only key (modify this)
        #print(f"Search item at index 0: {search_item}")
        found_track = search_item['name']
        print(f"Found track: {found_track}")
        # Making the found name and given name the same size
        found_track = found_track.lower()
        track_name = track_name.lower()
        #print(f"Found: {found_track}")
        #print(f"Original: {track_name}")
        # Checking if the found track name matches the given one
        if found_track == track_name:
            #print("Found track name:", found_track)
            track_uri = search_item['uri'] # gives track uri; a string            
            
            # Splitting the track_uri to retrieve the track ID
            track_uri = track_uri.split(':') # gives a dictionary
            track_id = track_uri[-1] # Retrieves the last element which is where the ID is
            print("found the track in ExtractTrackID, returning to GetSeveralAudioFeatures") # returning to geetseveralaudio features whenever we are using it in this case if not it just returns to wherever else we were
            return track_id # returns a string # good
        else:
            print("No match")
            return None

def SearchItems(access_token, search, search_type = [], artist_name = None, market = "US"):
    # Parameter meanings:
    # access_token: self explanatory
    # search: search query; can be album, artist, track, year, upc, tag, isrc, and genre, this is the name of whatever you are looking for
    # search_type: an array of strings (list) of items to search across, where you can add more than one; possible searches...
    # ... are track, artist, album, playlist, show, audiobook. Default is null if nothing is entered, which searches everything
    # artist_id: Used when needed to search for specific artist items (i.e., tracks, playlists, shows, etc); Default is none
    # market: country, default is US
    url = "https://api.spotify.com/v1/search"
    header = GetHeader(access_token)
    
    # Converting search_type to a list so we are able to search for each query
    print("\nIn SearchItems")
    print(f"artist name: {artist_name}")
    print(f"search type: {search_type}")
    search_list = search_type.split(',')
    for i in range(len(search_list)):
        search_list[i] = search_list[i].strip(" ")
    
    ran = False # To signify if a POST request has already been done
    count = 0
    # Checking what the search and search type is
    for i in search_list:    
        i = i.lower()
        print(f"what is i?: {i}")
        #print(search_list[i])
        if i == "artist": # Artist search
            query = f"?q={search}&type=artist&limit=1"
            
        elif i == "album": # Album search
            query = f"?q={search}&type=album&limit=5"
            
        elif i == "track": # Track search
            if artist_name: # If we are given an artist name, then we can search specifically within the artist's discography for the track
                # expand search to 20, enter a for loop to see if we cannot find the artistname/artist id(maybe do ID), then keep expanding the search
                #query = f"?q=track:{search} artist:{artist_name}&type=track&limit=5"
                #query = f"?q={search}&type=track&limit=5&artist_id={artist_id}"
                found = False # To signify whether we have found the desired track
                found_request = None # Holds the request we are searching for with the given artist and track name
                while (found != True):
                    query = f"?q={search}&type=track&limit=1" # MIGHT NEED TO SWITCH LIMIT IF i want to send in more than one item at a time (i dont want that right now)
                    query_url = url + query
                    request = POSTRequest(query_url, header)
                    #print(type(request)) # a dict
                    #print(request)
                    #print(request.keys())
                    #print("items:")
                    #print(request['tracks']['items'])
                    #print(type(request['tracks']['items']))
                    for i in request['tracks']['items']:
                        #print(f"at {i} in request['tracks']['items']")
                        for artist in i['artists']:
                            #print(f"at {artist} in {i}['artists']")
                            found_artist = artist['name']
                            found_artist = found_artist.lower()
                            print(f"found artist: {found_artist}")
                            print(f"given artist: {artist_name}")
                            print(f"track id key: {i['id']}")
                            print(f"track name key: {i['name']}")
                            print("requests and next:", request['tracks']['next'])
                            print(f"found artist type: {type(found_artist)}")
                            print(f"given artist type: {type(artist_name)}")
                            print(f"lowercase given_artist: {artist_name.lower()}")
                            if found_artist == artist_name.lower():
                                print("they are the same")
                                found = True
                                print()
                                found_request = request
                                print(f"\nFound request: ")
                                #print("hi here")
                                #print(found_request, "\n\n")
                                return request
                            else: # Meaning they are not equal
                                print("names not equal")
                                if 'next' in request['tracks']:
                                    print("here (if 'next' in request['tracks']")
                                    # never ends loop here
                ran = True # Tells us if we went through the loop and were able to get a result; this prevents the request from running again
                print("out of loop")
            else:
                query = f"?q={search}&type=track&limit=5"
                ran = False # If this is false then we can get a request (this happens when artist_name is not given)
                print("artist name did not pass through we went straight to 'else' in SearchItems")

def SearchByGenre(access_token, genre, year_range = None, market = "US", limit = 10, max_tracks = 50, offset = 0): # Searches by genre (for use in genre trend analysis)
    # Variable names:
        # genre: the genre we want to search for
        # year_range: the years we want to look for tracks in; default is none; a string
        # limit: number of searches to return (used to be 1)
        # max_tracks: maximum number of tracks we will obtain at a time (used to be 13 for testing)
        # offset: used for pagination; meaning if we have an offset of 0 and a limit of 3, then we are retrieving items 0-2, etc
        
        header = GetHeader(access_token)

        # Collect all the tracks and put it into a dataframe
        track_info = {} # For each individual track
        all_track_info = [] # To hold each individual track_info (a list of a dictionary)
        index = 0
        
        # Steps: 
            # 1. loop through data based on how many tracks we want
            # 2. with the obtained track data, which will be in the form of a dictionary, search for key attributes ...
            # ... that we can use to sort the tracks (i.e., track name, track ID, track ISRC, artist(s) name, etc),
            # ... store this information in a dictionary then move to a dataframe OR directly store into a dataframe
            # 3. get the audio features of each track and store it, in relation to its track ID (can do a check to see if ...
            # ... the artist name(s), track name, & ISRC match as well) in the dataframe/dictionary/whatever option ...
            # ... i did in #2
            
        while index < max_tracks: # Keep going through the loop until we reach the max amount of tracks (for filling in tracks)
        #while len(all_track_info) < max_tracks: # To account for new tracks that had to be generated
            # len(all_track_info) messes with correlation matrices (i.e., shows more even correlations, such as 1)
            #print(f"\ncurrent index value: {index}")    
            offset = limit * index # This is so we continue to get new searches each time the index is increased
            track_data = SearchGenreTracks(genre, limit, header, market, offset, year_range) # Call function to search for tracks of the given genre
            #print("\n", track_data, "\n") # good gives me all different songs
            #print(type(track_data))
            # for track data, call the function that gets the audio features, and put these audio features into a dataframe/dict
            # as well as the title, artists, release date, etc, basic info basically
            
            # Getting certain track information and storing it in a dictionary  # good works when limit goes up as well
            tracks = track_data['tracks']['items'] 
            for i in tracks:
                
                # Checking if we have a valid release date, for future data analysis functions
                release_date = i['album']['release_date']
                
                if len(release_date) != 10: # Skips adding the track which does not have a valid release date
                    continue
                
                track_info = {'track_name' : i['name'],
                               'artists' : [artist['name'] for artist in i['artists']],
                               'release_date': i['album']['release_date'],
                               'duration_ms' : i['duration_ms'],
                               'track_id' : i['id'],
                               'isrc' : i['external_ids']['isrc']}
                #print(f"\n\ncurrent tracks_info: {track_info}")
                all_track_info.append(track_info) # Adds the current track_info to the overall dictionary (a list)
                
                #if len(all_track_info) >= max_tracks: # Stops adding more tracks if all_track_info is full (mainly used when there is an invalid release date)
                 #   break
            index = index + 1
        # Printing all tracks and their info to know 
        print("release date access: ")
        for i in all_track_info:
            print(i['release_date'])
        print("\n\n")
        print("\nAll track information:")
        for i in all_track_info:
            print(f"\n{i}")
        print("\n\nnumber of tracks we have to analyze: ", len(all_track_info)) # good
        
        # Putting track names and artist name(s) in collected tracks into its own list for analysis
        """
        # dont need this im pretty sure
        all_track_names = []
        all_artist_names = []
        for i in all_track_info:
            all_track_names.append(i['track_name'])
            all_artist_names.append(i['artists'])
        print("all track names and their type:\n")
        for i in range(len(all_track_names)):
            print(all_track_names[i])
            print(type(all_track_names[i]))
        """
        return all_track_info

def GetSeveralAudioFeatures(track_names, artist_name = None): # Gets lots of information about a song (danceability, etc whatever) <-- use this for data for now
    endpoint = "https://api.spotify.com/v1/audio-features"
    header = GetHeader(access_token)
    print("In GetSeveralAudioFeatures function")
    print(f"Given artist_name: {artist_name}")
    print(f"given track name/ID: {track_names}")
    #print(type(track_names))
    # Checking if we have more than one track given (in the form of a list usually), then convert the list to a string
    #print()
    if type(track_names) == list:
        print("its a list")
        #track_names = ','.join(track_names)
        track_names = ',,'.join(track_names) # this should work for most cases if there is a , in a song title
        #print(track_names)
        #print(type(track_names))
        track_list = track_names.split(',,')
        #print("splitted list:")
        #print(track_list)
    else: # Fix this because its separating my tracks if they have ,
        #print(type(track_names))
        #print(track_names)
        # Splitting track_names and removing any whitespace
        track_list = track_names.split(',,') # list
        #print(track_list) 
        #print(type(track_list))
    
    for i in range(len(track_list)):
        track_list[i] = track_list[i].strip()
    #print(type(track_list))
    track_ids = [] # To store track IDs
    
    # DO SOMETHING IN THIS FUNCTION TO CHECK IF WE ARE GIVEN IDS (i.e., do something to check if track_names is a list of IDs)
    # Steps:
        # 1. put a check right here to see if every individual track (so go through a for loop like the one below) is ...
        # ... 22 characters long AND is a mix of letters and numbers (alphanumeric (v string.isalnum()))
        # 2. if the string is alphanumeric and 22 characters long, then continue on, but still add each track ID to ...
        # ... the track ID list so we can continue the analysis (which should still work after this)
        # 3. else if the string is not alphanumeric and 22 characters long exactly, then we continue the for loop below ...
        # ... inside the else statement, then we continue going out of the loop from there
        # --> maybe put steps 1 and 2 in a different function to evaluate it
    
    # Seeing if we are given IDs instead of track names # very good
    for track in track_list:
        if len(track) == 22 and track.isalnum() == True: # If the length of the track we are on is 22 characters AND alphanumeric, then its a track ID
            track_ids.append(track) # Adds the current track we are on to the track_ids list for further analysis
        else: # Its probably a track name and not an ID unles we got really unlucky
            track_id = ExtractTrackID(track, artist_name)
            track_ids.append(track_id)
        
    """  # probably remove even though it was good we did it above now
    # Getting the ID of each track in given track_list for analysis later on
    for track_name in track_list: 
        print(f"Track name: {track_name}")
        track_id = ExtractTrackID(track_name, artist_name) # probably add artist id here
        track_ids.append(track_id)
    """
    #print(track_ids) # good (list)

    # Creating URL using track_ids and getting the request
    track_ids_string = ",".join(track_ids)
    url = endpoint + "?ids=" + track_ids_string
    #print(url) # good
    request = POSTRequest(url, header)
    print(type(request))
    print("\n", request)
    print(request.keys())
    print(request['audio_features'][0]['uri']) # first song in the request
    
    num_songs = len(request['audio_features']) # length of however many arguments we have
    
    """
    # Analyzing tracks individually
    # Logic: 
    # 1. go through a for loop
    # 2. filter the dictionary by its URI/ID (maybe we should have stored the ID and name of the track in a dictionary )
    #   -> if we dont do the dictionary method, just call/make a new function to retrieve the name based on ID
    # 3. from here, we can separate it based on its features (all in a new dictionary, associated with its name/ID)
    # 4. from here, do another function where all this data can be returned to (probably within a for loop in the other function)
    # ... for data analysis (graphs, etc)
    #   -> maybe make a function where we can take whole albums and their individual track IDs (or us a combo of functions we have which might be doable)
    #   ... in order to extract their info, and maybe we can summarize the album components (danceability, energy, etc)
    # do something else to this data maybe for top albums worldwide or an entire artist's discography    
    #print(len(track_ids)) # should be the same length as num_songs
    # use track IDs here to compare
    """
    track_info = {} # To store track information
    #print(type(request))
    # Getting track IDs
    for i in range(num_songs):
        current_track_id = request['audio_features'][i]['id']
        #print(current_track_id)
        #print(request['audio_features'][i]['id'])   
        if current_track_id == track_ids[i]: # If they are the same ID, which it should be, then we can add this to the track_info dictionary
            """    
            print(f"Song title: {track_list[i]}")
            #print(request['audio_features'][i], "\n\n") # good
            #track_info[current_track_id] = request['audio_features'][i]
            #print(track_info[current_track_id], "\n\n")
            """
            attributes = request['audio_features'][i]
            
            # Creating track info dictionary where we can look for tracks based on their ID (probably call the function if needed in other ways)
            track_info[track_list[i]] = {
                "Attributes" : {key: attributes[key] for key in attributes.keys()}
                }
    """        
    print(track_info.keys(), "\n\n") # The name of the track is the key
    #print(track_info['imperfect for you']['Attributes'].keys()) # Gives me the keys inside attributes # good
    #print(track_info[track_list[0]]['Attributes']['loudness']) # good
    #print(track_info[0])
    #print("\n\n", track_info['imperfect for you'])
    """
    return track_info # a dictionary (use this in another function to sum each value up and give an average for the album itself)

# Data analysis helper functions
def MostCorrelatedValues(correlation_matrix, parameters, top_values = 3):
    # Variable meanings:
    # correlation_matrix: the correlation matrix that we send in to find the most correlated values
    # parameters: the parameter(s) we are finding the most correlated values for; a list
    # top_values: the amount of values to check that are the most correlated wrt to each parameter (default is 3)
    #print("correlation matrix:")
    #print(correlation_matrix)
    top_correlated_values = {}
    # Iterating through the parameters list given to analyze the most correlated values
    for i in parameters:
        series = correlation_matrix[i] # Creating a series to analyze at the current i value in the parameter list
        sorted_series = series.sort_values(ascending = False)
        #print(sorted_series)
        # Sorting the series by absolute value, so we can identify the most correlated values easier
        abs_value_series = series.abs().sort_values(ascending = False)
        #print(abs_value_series)
        # Extracting most correlated values by name at the current parameter (i.e., we are storing the names for now)
        top_correlated_parameters = abs_value_series.index[1:top_values + 1] # Starting at one index over to not account for the actual value we are testing
        #print(top_correlated_parameters) # good
        #print(type(top_correlated_parameters))
        # Extracting most correlated values based on top_values at the current parameter
        top_correlated_values[i] = {} # Creating a dictionary within the dictionary to correctly store highly correlated values to the right parameter
        for j in top_correlated_parameters:
            #print(f"j: {j}")
            correlation_value = series[j] # good
            #print(f"series at {j}: {correlation_value}") 
            top_correlated_values[i][j] = correlation_value # Storing the proper correlation value at the current parameter
    #print(top_correlated_values) # pretty sure this is good

    return top_correlated_values # Returns a dictionary

def Stacked_LSTM_Model(input_shape): # To build a stacked (multidimensional) LSTM model based on trained data
    # Parameter explanation:
        # input_shape: The sequence length (number of steps we go back to) and the number of parameters/features, obtained from X_train data
    
    model = Sequential()
    model.add(LSTM(10, activation = 'relu', return_sequences = True, input_shape = input_shape)) 
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mse')    
    return model

def GenreTrendAnalysis(genre_tracks, genre_decade):
    # Variable meanings:
    # genre_tracks: a list holding dictionaries of track features such as track name, duration (ms), ID, release date, etc
    # genre_decade: a string that tells us what genre and decade we are looking at (mainly for use in visualization)
    print(genre_tracks)
    # Creating a dataframe to store all information
    columns = ['Track Name', 'Artist(s)', 'Release date', 'Duration (ms)', 'Danceability', 'Energy', 'Key', 'Loudness', 
               'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness','Liveness', 'Valence', 
               'Tempo', 'Track ID', 'ISRC'] # maybe add a duration in mm:ss
    df = pd.DataFrame(columns = columns)
    all_audio_features = []
    
    # Getting audio features of each track
    for track in genre_tracks: # good i think (just format the audio features better for printing if needed)
        track_name = track['track_name'] # Gets the current track name at current index
        #artist_name = track['artists'][0] # Gets the artist for the current track at the current index (only the first one, which is usually the primary)
        track_artists = ", ".join(track['artists'])
        track_release_date = track['release_date']
        track_isrc = track['isrc']
        track_id = track['track_id'] # Gets current track ID at current index
        track_duration = track['duration_ms']
        track_features = GetSeveralAudioFeatures(track_id) # Gets audio features of current tracks (dont have to send in artist); a dictionary with the key being the track ID
        
        # Extracting audio features from the track so we can organize the keys a bit better
        audio_features = track_features.get(track_id).get('Attributes')
        #print(audio_features.keys())
        
        #print(track_features[track_id]['Attributes'].keys())
        #all_audio_features.append(track_features) # a list

        track_data = {'Track Name' : track_name,
                      'Artist(s)' : track_artists,
                      'Release date' : track_release_date,
                      'Duration (ms)' : track_duration,
                      'Danceability' : audio_features['danceability'],
                      'Energy' : audio_features['energy'],
                      'Key' : audio_features['key'],
                      'Loudness' : audio_features['loudness'],
                      'Mode' : audio_features['mode'],
                      'Speechiness' : audio_features['speechiness'],
                      'Acousticness' : audio_features['acousticness'],
                      'Instrumentalness' : audio_features['instrumentalness'],
                      'Liveness' : audio_features['liveness'],
                      'Valence' : audio_features['valence'],
                      'Tempo' : audio_features['tempo'],
                      'Track ID' : track_id,
                      'ISRC' : track_isrc} # A dictionary where we can hold the corresponding track name with its data
        #track_data.update(track_features) # Updating the current track data with its corresponding features
        
        # Adding track to the data frame
        #df = df.concat([df, pd.DataFrame([track_data])], ignore_index = True)
        df = pd.concat([df, pd.DataFrame([track_data])], ignore_index=True)

    
    """
    # Printing tracks and features
    for feature in all_audio_features: # idk do something here but it prints in general doing print() i just dont want to bother reading
        PrintAudioFeatures(feature)
    """
    """
    for column in df.columns:
        print(f"Column: {column}")
        print(df[column].to_string(index=False))
        print("\n")
    """

    # Correlation Analysis 
    # Parameters we will be analyzing: danceability, energy, key, loudness, mode, speechiness, acousticness, ...
    # ... instrumentalness, liveness, valence, tempo, time_signature
    # Possiblity: duration (in ms) to see if there is some other correlation between track length and other features, but 
    # ... probably not
    
    analyzed_columns = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 
                        'Instrumentalness', 'Liveness', 'Valence', 'Tempo'] # Columns we want to analyze in the dataframe
    
    df_analyzed = df[analyzed_columns] # The data frame but with only the columns we want to analyze
    """
    print("Analyzed df")
    for column in df_analyzed.columns:
        print(f"Column: {column}")
        print(df_analyzed[column].to_string(index=False))
        print("\n")
    """
    # Kendall's Tau
    # We will be conducting Kendall's Tau by pairing each parameter with each other and then comparing the results
    # Then we can consider other methods of correlation analysis (multivariate, maybe a PCA) to see if there are any chanegs
     
    kendalls_tau = df_analyzed.corr(method = 'kendall')
    
    print("Kendall's Tau (with paired variables")
    print(kendalls_tau) # good an 11 x 11 matrix 
    # sort the matrix in order next
    
    # Visualizing the results of Kendall's Tau using a heatmap
    kendalls_heatmap = sns.heatmap(kendalls_tau, vmin = -1, vmax = 1, cmap = 'rocket_r', center = 0, cbar = True, annot = True) # vmin = -1 for least correlated, opposite for vmax
    plt.title(f"Kendall's Tau Heatmap for {genre_decade}", size = 15, pad = 8)
    plt.show() # Good but maybe change the colormap to something better
    
    # Spearmans's Correlation
    spearman = df_analyzed.corr(method = 'spearman')
    print("Spearmans's Correlation")
    print(spearman)
    spearmans_heatmap = sns.heatmap(spearman, vmin = -1, vmax = 1, center = 0, cmap = 'crest', cbar = True, annot = True)
    plt.title(f"Spearman's Correlation Heatmap for {genre_decade}", size = 15, pad = 10)
    plt.show()    

    #print(df.head())
    #print(df.columns)
    #print(df['Release date'])
    # ********** Time Series Analysis **********
    # BEFORE DOING THIS RUN IN LARGE AMOUNTS OF DATA AND THEN NARROW DOWN THE DECADES (5): end at present day
    # so do this first:
        # filter in songs of a genre from the decade range(s) we want, separate into its own data frame; possibly do the bigger hits
        # do the analysis on EACH decade (spearmans and kendalls)
        # do the time series analysis on EACH decade (this will probably be done by the function by defauly anyway)
    # OR TO LOOK AT THE MOST POPULAR TRACKS, SEARCH FOR PLAYLISTS (i.e., 80s pop playlist) and analyze
    # OR SEARCH BY CATEGORY ID (if they even separate it by genre idk maybe use keywords)
    
    # Using Kendall's Tau and Spearman's Correlation, we can see the most highly correlated variables with respect to ...
    # ... each variable. We will choose three variables to analyze for simplicity, and then choose the three most ...
    # ... correlated variables for each of those variables. 
    
    # Parameters chosen for the three genres we selected (pop, rap, country): danceability, energy, loudness
    
    # Correlated variables wrt to each genre and decade (put this section in main/readme); but do this when you read a lot more data in
    # 2010s pop
    # Most highly correlated variables with danceability: valence, acousticness, and mode
    # Most highly correlated variables with energy: loudness, liveness, speechiness
    # Most highly correlated variables with loudness: valence, energy, danceability/liveness (tied)
    
    # ***** Then, using the most correlated variables (probably extract them from the matrix, deal with when we have ties ...
    # ... between several parameters (just include them all basically into the analysis), conduct a time series analysis ...
    # ... where we visualize the trends and infer from there (do we need to see if its stationary?)
    
    """
    # We can/might send in the three variables (depending on genre) or just hard code them in to analyze each component ...
    # ... although depending on the genre, the most correlated variables will be different, so possibly send them in
    
    # Parameters chosen (2010s pop): danceability, energy, instrumentalness, 
    # Most highly correlated variables with danceability: valence, acousticness, and mode
    # Most highly correlated variables with energy: loudness, liveness, speechiness
    # Most highly correlated variables with instrumentalness: tempo, liveness, acousticness
    
    # ... do this for all genres and their decades (these comments will be in main in new program)
    """
    
    # Plan for time series analysis:
        # Its being done depending on the years being sent in, so no need for an extra dataframe
        # Now that we have the years done, conduct the time series analysis
        # Steps:
            # 1. Do an EDA (i.e. correlation matrices)
            # 2. Visualize data
            # 3. Check if data is stationary (needed for forecasting if i wanted to use ARIMA/SARIMA (idk about lstm))
            # 4. Identify patters based on visualization, correlation matrix, and 
    # 1. Exploratory data analysis (EDA)
    # We want to visualize how music has changed within the current decade
    
    # Extracting most highly correlated variables wrt to the three parameters we chose
    parameters = ['Danceability', 'Energy', 'Loudness']
    spearmans_mostcorrelated = MostCorrelatedValues(spearman, parameters, 4) # a dictionary of dictionaries
    #kendalls_mostcorrelated = MostCorrelatedValues(kendalls_tau, parameters, 4) # a dictionary of dictionaries
    #print(spearmans_mostcorrelated)
    #print("release date")
    #print(df['Release date']) # PUT ERROR handling at release date (probably when reading in the data frame) to either ...
    # ... 1. not count any dates that dont have a full date (i.e., just the year)
    # 2. put a default date (i.e., 1/1/XXXX) as the date if no full date is available

    # 2. Visualize data
    # We can visualize the data using smooth plot lines, scatter plots, and area charts 
    #print(df.keys())
    
    # MIGHT NEED TO REDO DATETIME FORMAT, NOT INCLUDE IT FOR LABELING SINCE ITS MESSING THINGS UP (code isnt filtering the years in SearchByGenre correctly)
    # or just keep regenerating until you get a bunch of valid years but idk
    # Converting release date to datetime format, sorting it, then obtaining the years
    df['Release date'] = pd.to_datetime(df['Release date']) # made graph all weird and squiggly everywhere # good now i think
    df = df.sort_values(by='Release date')
    #df['Year'] = df['Release date'].dt.year
    
    # Plotting each parameter wrt to time
    # These are line graphs
    for i in parameters: # time vs parameter
        plt.plot(df['Release date'], df[i]) # somehow make it so the labels at the bottom are start_date and end_date, but still use release date data
        plt.xlabel('Year')
        plt.ylabel(i)
        plt.title(f'Time vs {i} for {genre_decade}')
        plt.show()
        #print(df['Loudness']) # check why loudness is all negative but apparently they are all negative anyway # its fine the max is 0 while minimum is -60
    
    """
    # Plotting the most correlated variables wrt to the parameter and time
    # Using Spearman's results 
    for key, value in spearmans_mostcorrelated.items():
        # Obtaining the correlated variables inside the dictionary
        correlated_values = spearmans_mostcorrelated[key]
        for parameter, correlation_value in value.items():
            print("correlation value: ", correlation_value)
            plt.plot(df['Release date'], df[parameter], label=f"{parameter} (Correlation: {correlation_value:.2f})")
    
        plt.plot(df['Release date'], df[key]) # somehow make it so the labels at the bottom are start_date and end_date, but still use release date data
        plt.xlabel('Year')
        plt.ylabel(i)
        plt.title(f'Time vs {key} and Most Correlated Variables (Spearman\'s)')
        plt.legend()
        plt.show()
    
    # Using Kendall's Tau results
    """
    
    # 3. Check if data is stationary
    # To check if the data is stationary, we need to use a statistical test and check the p-value 
    # We will be using the Augmented Dickey-Fuller (ADF) and the Kwiatkowski–Phillips–Schmidt–Shin (SPSS) tests in conjunction with each other
    # We will also be checking the rolling mean and standard deviation of the data during the decade
    
    # Checking the rolling mean and standard deviation for each parameter
    window = 12 # Checking over a window of 1 year
    
    # Error checking for forecasting model
    """
    fake_p = 0.1
    use_fake_p = True
    """
    
    for i in parameters:
        print(f"{i} rolling mean and rolling std: ")
        rolling_mean = df[i].rolling(window = window).mean()
        rolling_std = df[i].rolling(window).std()
        
        # Dropping NaN values (usually obtained when the data fed in is less than the window we are looking at)
        rolling_mean = rolling_mean.dropna()
        rolling_std = rolling_std.dropna()
        print(rolling_mean)
        print(rolling_std)
        # drop NaN stats and add +12 to the songs we call to account for these values
        
    # Using ADF to check if data is stationary
    # p value needs to be p < 0.05 for time series to be stationary
    stationary_results = {} # To hold boolean and p-values values which check if the parameter's time series is stationary or not
    stationary = False # To know if the time series is stationary or not; default is false
    for i in parameters: 
        adf_result = adfuller(df[i]) # A tuple type
        print(f"{i} ADF result: ")
        print(adf_result)
        print(type(adf_result))
        p_value = adf_result[1] # Accessing the p value of the current parameter
        print(f"{i} p-value: {p_value}")
     
        # Checking if time series is stationary
        # Using p-value
        if p_value < 0.05: # Time series is stationary
            print(f"Time series for {i} is stationary")
            stationary = True
        else: # Time series is not stationary
            print(f"Time series for {i} is not stationary")
            stationary = False
            
        stationary_results[i] = (stationary, p_value) # The corresponding value for the dictionary is a tuple 
        print("\n")
    
    # Deciphering the ADF result, in order: 
        # Test statistic
        # p-value
        # Number of lags (lags meaning a previous observation in the data used in the time series)
        # Number of observations after lagged difference; used for critical values as well
        # Critical values are in the dictionary response, where we test at different significance levels (the probability ...
        # ... of the event occuring (in percent))
        # AIC value, which measures how well the model fits with the data (lower values = better model fit)        

    # From here, interpret results of ADF test
    # If a time series is stationary, then we know that the parameter we are on does not change over time
    # If a time series is not stationary, then the parameter we are on does change over time wrt to 
    
    """
    # Using SPSS to check if data is stationary (maybe dont use this i dont remember the module)
    # p value needs to be p > 0.05 for time series to be stationary
    #for i in parameters:
     #   spss_result = 
    """
    # 4. Forming conclusions and identifying patterns (probably no code here just analysis)
    # Create scatter plots, line graphs, etc to see how different features interact over time. 
    # maybe do this for the whole 50 years as well (so just change the decade range argument when calling the function)
    
    # done with all of the above just need to feed in all the data after this
    
    # ********** Forecasting Models ***********
    # This will be used to predict possible genre evolution (probably a LSTM)
    # Can use ADF to help with forecasting
    
    # If the time series data for the parameter is stationary, then we can use it directly to train and test the data
    # Otherwise, we need to transform/modify the data somehow in order to use it
    
    #print("\n\nHandling when data is stationary or not for LSTM\n\n")
    # Handling when data is stationary or not stationary
    for parameter, (stationary_value, p_value) in stationary_results.items(): # Going through each parameter and its tuple in the dictionary
        #stationary_value = parameter[1][0] # Accesses stationary value in stationary_results; for easy access
        if stationary_value == True: # Meaning its stationary
        # If the data was stationary, then we can directly train 
            print(f"{parameter} was stationary")            
        else: # Meaning its false
            print(df[parameter])
            print(f"{parameter} data is not stationary, differencing data...")
            #print("If we start looping/breaking way after anything else we are in the forecasting models loop")
        # If the data was not stationary, then we need to difference the data in order to make it stationary
            new_stationary = False
            while new_stationary == False:
                print("In loop")
                #print("in do while")
                df[f'{parameter}_differenced'] = df[parameter].diff() # Drops any NaN values and differences data
                df[f'{parameter}_differenced'].dropna(inplace=True)
                #print(f"\n{parameter} differenced: ")
                #print(df[f'{parameter}_differenced'])
                # nan value is still in place for 1 value at least, possibly replace those with 0s
                # Model Sensitivity Testing: Evaluate your LSTM model's performance with different approaches to ...
                # ... handling NaN values. Compare results between filling with 0, using mean/median imputation, or ...
                # ... excluding NaN values altogether (where feasible).
                #print("hello, seeing what NaN values we have")
                # Counting if we have NaN values
                nan_count = df[f'{parameter}_differenced'].isna().sum()
                #print(f"nan count: {nan_count}")
                if nan_count > 0: # Meaning we have NaN values
                    #print("filling NaN values")
                    df[f'{parameter}_differenced'].fillna(0.00, inplace = True) # Filling NaN values with 0.00
                    #df[f'{parameter}_differenced'].fillna(df[f'{parameter}_differenced'].mean(), inplace=True) # Filling with mean
                    #print("done filling in")
                #print(df[f'{parameter}_differenced'])
                # Checking if the data was made to be stationary, and if it wasnt, we keep looping until it is
                differenced_adf = adfuller(df[f'{parameter}_differenced'])
                print(f"differenced ADF: {differenced_adf}")
                differenced_p_value = differenced_adf[1]
                
                if differenced_p_value < 0.05: # Then it is stationary
                    print(f"Data for {parameter} is stationary after differencing")
                    new_stationary = True # Exits out of loop
                elif differenced_p_value >= 0.05: # Not stationary
                    new_stationary = False # Continues the loop
                    print(f"Data for {parameter} is not stationary after the differencing, using log transformation...")
                    # do another method if this doesnt work
                    df[f'{parameter}_log_transform'] = np.log1p(df[parameter]) # Log transformed data
                    # Note that log1p deals with our smaller values, so its a bit more accurate in this case
                    #print(df[f'{parameter}_log_transform'])
                    log_adf = adfuller(df[f'{parameter}_log_transform'])
                    log_p_value = log_adf[1]
                    #print(log_p_value)
                    if log_p_value < 0.05: # Then it is stationary
                        print(f"Data for {parameter} is stationary after log transformation")
                        new_stationary = True
                    elif log_p_value >= 0.05: # Still not stationary
                        print(f"Data for {parameter} is not stationary after log transformation, using non-stationary data...")
                        #new_stationary = False
                        break # Breaking out of the loop to use the non-stationary data if all else has failed
    
    # After the data is made stationary, we can start creatng the LSTM
    
    # Sequencing data (finding patterns)
    """
    # Extracting year from genre_decade string
    decade = genre_decade[:4] # Takes the first four characters, which is all we need (it will always be in that format)
    print(decade)
    # Since we are analyzing music from 1970 (4) - 2019 (2024), we can hardcode that data depending on the given genre
    # maybe we really dont need this in the end
    if decade == "1970": # 1970s music
        start_date = 1970
        end_date = 1979
    elif decade == "1980": # 1980s music
        start_date = 1980
        end_date = 1989
    elif decade == "1990": # 1990s music
        start_date = 1990
        end_date = 1999
    elif decade == "2000": # 2000s music
        start_date = 2000
        end_date = 2009
    elif decade == "2010": # 2010s music
        start_date = 2010
        end_date = 2019
    else: # Invalid decade given, default to 2010
        start_date = 2010
        end_date = 2019
    """
    sequence_data = {}
    sequence_length = 10 # maybe change to 50, or take in decade and see idk maybe hard-code this in the future for graphs
    for i in parameters:
        # X data is the parameters, Y data is what we are predicting (i.e., how does parameter change wrt time)
        X_data = [] #df[parameters].values
        Y_data = [] #df[i].values
        #print(type(X_data)) # numpy nondimensional array
        sequence_data[i] = Y_data
        
        # Each sequence in X_data holds the entire history of all the parameters up until the time we want to predict
        for j in range(len(df) - sequence_length): # Traversing the entire dataframe minus the length of our sequence ...
        # ... to make sure we have enough values and to avoid any NaN or out of bounds values that may occur when creating the sequence
            X_sequence = df[parameters].iloc[j:j+sequence_length].values # Gets a slice of data for all parameters from j to j + the length of the sequence, and converts it into a numpy array (values)         
            Y_values = df[i].iloc[j+sequence_length] # Gets data at the current parameter (i) and selects the data located at the current j + the sequence length; what we will be using to predict
            
            # Appending data to X_data and Y_data for use in training and testing the model
            X_data.append(X_sequence)
            Y_data.append(Y_values)
        
        # Converting X_data and Y_data into numpy arrays for easier use in the LSTM
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)
    
        # Adding the data to sequence_data to store it for easy access wrt each parameter
        sequence_data[i] = (X_data, Y_data) # stores X and Y data as a tuple for each parameter
    
    # Checking shape to ensure the size of the data is correct
    #print(X_data.shape)  # Should be (num_sequences, sequence_length, num_parameters)
    #print(Y_data.shape)  # Should be (num_sequences)
    
    # Splitting data into training and testing data
    train_test_split_data = {} # Holds the training and testing data for each parameter
    for i in parameters: 
        X_test, X_train, Y_test, Y_train = train_test_split(sequence_data[i][0], sequence_data[i][1], test_size = 0.2) # 20% of our data will be testing data
        train_test_split_data[i] = (X_test, X_train, Y_test, Y_train) # Holds the testing and training data for each parameter as a tuple
        
    # Scaling data
    scalers = {}
    for i in parameters:
        scaler = MinMaxScaler() # Starting at a range from 0-1,, can also do -1-1
        X_data, Y_data = sequence_data[i] # Getting the sequences (X) and data we want to analyze (Y) from sequence_data at i
        
        # Scaling Y_data to be a 2D array and putting it back into sequqnece_data
        Y_data_scaled = scaler.fit_transform(Y_data.reshape(-1, 1)) # The -1 means as many rows as we need to fit all of our data
        
        sequence_data[i] = (X_data, Y_data_scaled) # Adding the scaled values to our scalers dictionary
        
        scalers[i] = scaler
    
    #print(Y_data_scaled.shape) # Checking shape (should be (num_sequences, 1))
    #print(X_train.shape)
    #print(sequence_data.items())
    
    # Creating the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    """
    lstm_model = Sequential()
    lstm_model.add(LSTM(10, activation = 'relu', return_sequences = True, input_shape = input_shape)) # X_train.shape[1] and X_train.shape[2] are the sequence length (number of steps we go back to) and the number of parameters/features
    lstm_model.add(LSTM(50, activation='relu'))
    lstm_model.add(Dense(1)) 
    lstm_model.compile(optimizer='adam', loss='mse') 
    """
    lstm_model = Stacked_LSTM_Model(input_shape)
    # Model fitting
    for i in parameters:
        # Retrieving testing and training data from the current parameter
        X_train, X_test, Y_train, Y_test = train_test_split_data[i]
        
        lstm_model.fit(X_train, Y_train, epochs = 100, verbose = 0, validation_data = (X_test, Y_test))
        # Epoch means how many times we will work through the trained data set, while verbose gives the output of the data (0 meaning little to no, 2 being many details)
        loss = lstm_model.evaluate(X_test, Y_test, verbose=0) # Tells us how well the LSTM model can predict the outcome (i.e., how well does it match the Y data)
        
        # Higher values = the LSTM may not be able to predict the outcome as well
        # Lower values (near 0) = the LSTM can predict the outcome well
        print("Calculating loss values to determine how good of a fit the data was")
        print(f'Parameter: {i}, Test Loss: {loss}')

# Main Program

# Client ID and client secret
redirect_uri = "http://localhost:3000"
client_id = "c04466583c2e444d8d60c3bf6e7a7889"
client_secret = "6074f9e4949a4d058f1a616ce7304247"
scopes = "user-top-read" # probably add more to the scopes when needed, will need to change GetToken function in this case
access_token = GetToken(redirect_uri, client_id, client_secret, scopes)



# GENRE TREND ANALYSIS
# GOALS: 
# 1. Conduct a time series analysis on a select few music genres (pop, rock, country, r&b, rap, etc) across 
# 5 decades (pick a range)
# 2. do a correlation analysis (Spearman's Rank and Kendall's Tau) to identify the highest correlation among audio 
# features in each genre
# 3. Train and test data to create a forecasting model to predict genre evolution
# 4. Visualize data

# FUNCTIONS USED (for github): POSTRequest, GetHeader, GetToken, GetRefreshToken, GetHeader, SearchByGenre, 

# Genres picked (do 3 minimum): pop, r&b, country (maybe rap instead of r&b)

decades = ["1970-1979", "1980-1989", "1990-1999", "2000-2009", "2010-2019"]

# GENRE: COUNTRY
# Searching Spotify for 'country'
"""
print("\n\n**********\nCOUNTRY SEARCH\n**********")
genre = "country"

for decade in decades:
    year_range = decade
    print(f"{decade} COUNTRY ANALYSIS")
    country_search = SearchByGenre(access_token, genre, year_range = year_range)
    
    # Extracting the year for the decade (mainly for naming/listing)
    
    # Conducting a genre analysis 
    if type(country_search) is None:
        print("country_search returned nothing")
    else:
        print("country_search search did return something, doing genretrendanalysis")
        country_genre_trend_analysis = GenreTrendAnalysis(country_search, "{decade} Country")
"""
# GENRE: RAP
# Searching Spotify for 'rap'
"""
print("\n\n**********\nRAP SEARCH\n**********")
genre = "rap"

for decade in decades:
    year_range = decade
    print(f"{decade} RAP ANALYSIS")
    rap_search = SearchByGenre(access_token, genre, year_range = year_range)
    
    # Extracting the year for the decade (mainly for naming/listing)
    
    # Conducting a genre analysis 
    if type(rap_search) is None:
        print("rap_search returned nothing")
    else:
        print("rap_search search did return something, doing genretrendanalysis")
        rap_genre_trend_analysis = GenreTrendAnalysis(rap_search, "{decade} Rap")
"""
# GENRE: POP
# Searching Spotify for 'pop'
print("\n\n**********\nPOP SEARCH\n**********")
genre = "pop"
"""
for decade in decades:
    year_range = decade
    print(f"{decade} POP ANALYSIS")
    pop_search = SearchByGenre(access_token, genre, year_range = year_range)
    
    # Extracting the year for the decade (mainly for naming/listing)
    
    # Conducting a genre analysis 
    if type(pop_search) is None:
        print("pop_search returned nothing")
    else:
        print("pop_search search did return something, doing genretrendanalysis")
        pop_genre_trend_analysis = GenreTrendAnalysis(pop_search, "{decade} Pop")

"""
# pop 2010s, 2000s, 1990s work
year_range = "2010-2019"
pop_search_2010s = SearchByGenre(access_token, genre, year_range=year_range) # Returns a list (this is for any song released, not most popular ones)
#print(f"pop search: {pop_search_2010s}")
#print(f"pop search type: {type(pop_search_2010s)}")

# Conducting a genre analysis 
if type(pop_search_2010s) is None:
    print("pop_search_2010s returned nothing")
else:
    print("pop_search_2010s search did return something, doing genretrendanalysis")
    pop_genre_trend_analysis_2010s = GenreTrendAnalysis(pop_search_2010s, "2010s Pop")
    
    
    
    
    
