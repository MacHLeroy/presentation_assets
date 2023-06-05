"""
Dashboard.py
Baseball Dashboard by MacKenzye Leroy

This script allows the user to launch a baseball daushboard by running:

'streamlit run Dashboard.py' 

in the terminal.

Requires: 'YearlyResultsMaster.csv', 'LeagueGameResults.csv', 'PostSeasonStartDates.csv' in the same directory

All of which can be found on GitHub (github.com/MacHLeroy)

"""

# ----Imports------------------------------------
from turtle import title
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import streamlit as st
import plotly.express as px

# ----Set page configurations/defaults--------------
st.set_page_config(layout="wide")
WIDE_WIDTH = 1250
HALF_WIDTH = WIDE_WIDTH/2 -100
PLOT_HEIGHT = 450
TABLE_HEIGHT = 750



#Teams often change names, so we need a unique franchise identifier for each. The dictionary below maps all team names to a unique Franchise ID
#Franchise ID wil be referred to as FranID throughout this script

TEAMS_ID_DICT= {'Tampa Bay Devil Rays':'TBD', 'Tampa Bay Rays': 'TBD', 'Florida Marlins': 'FLA', 'Miami Marlins':'FLA',
                      'Montreal Expos':'WSN', 'Washington Nationals':'WSN',  'Seattle Pilots':'MIL', 'Milwaukee Brewers':'MIL',
                      'Houston Colt .45s':'HOU', 'Houston Astros':'HOU', 'Washington Senators':'MIN', 'Minnesota Twins': 'MIN',
                      'California Angels':'ANA','Anaheim Angels':'ANA', 'LA Angels of Anaheim':'ANA', 'Los Angeles Angels':'ANA', 
                      'Philadelphia Athletics':'OAK', 'Kansas City Athletics':'OAK', 'Oakland Athletics':'OAK', 'Cleveland Blues':'CLE',
                      'Baltimore Orioles':'BAL','St. Louis Browns':'BAL', 'Cleveland Indians':'CLE', 'Cleveland Naps':'CLE', 
                      'Boston Red Sox':'BOS', 'Boston Americans':'BOS', 'Cincinnati Reds':'CIN','Cincinnati Redlegs':'CIN',
                      'New York Yankees':'NYY', 'New York Highlanders':'NYY', 'Chicago Cubs':'CHC', 'Chicago Orphans':'CHC',
                      'Los Angeles Dodgers':'LAD', 'Brooklyn Superbas':'LAD','Brooklyn Dodgers':'LAD', 'Brooklyn Robins':'LAD',
                      'San Francisco Giants':'SFG', 'New York Giants':'SFG', 'New York Mets':'NYM', 'Atlanta Braves':'ATL', 
                      'Milwaukee Braves':'ATL', 'Boston Braves':'ATL', 'Boston Doves':'ATL', 'Boston Beaneaters':'ATL', 
                      'Boston Bees':'ATL', 'Boston Rustlers':'ATL',  'Pittsburgh Pirates':'PIT', 'Philadelphia Phillies':'PHI',
                      'Chicago White Sox':'CHW', 'Detroit Tigers':'DET', 'Texas Rangers':'TEX', 'Kansas City Royals':'KCR', 
                      'San Diego Padres':'SDP', 'Arizona Diamondbacks':'ARI', 'Seattle Mariners':'SEA',
                      'Toronto Blue Jays':'TOR', 'Colorado Rockies':'COL', 'St. Louis Cardinals':'STL' ,
                      'Baltimore Terrapins':'FedBAL', 'St. Louis Terriers':'FedSTL','Brooklyn Tip-Tops':'FedBRK',
                      'Pittsburgh Rebels': 'FedPIT', 'Kansas City Packers':'FedKCP', 'Indianapolis Hoosiers':'FedINDNEW',
                      'Newark Pepper':'FedINDNEW', 'Buffalo Buffeds':'FedBUF', 'Buffalo Blues':'FedBUF', 'Chicago Whales':'FedCHI',
                      'Chicago Chi-Feds':'FedCHI', 'Cleveland Bronchos':'FedCLE'  
                     }


#current MLB team names
CURRENT_TEAMS = ['Cincinnati Reds',  'Pittsburgh Pirates', 'Philadelphia Phillies',
                'Chicago White Sox', 'Detroit Tigers', 'Baltimore Orioles', 
                'Milwaukee Brewers', 'Chicago Cubs',  'Boston Red Sox', 
                'New York Yankees', 'Cleveland Indians', 'San Francisco Giants', 
                'Los Angeles Dodgers', 'Minnesota Twins', 'New York Mets', 
                'Houston Astros',  'Atlanta Braves', 'Oakland Athletics', 
                'Kansas City Royals', 'San Diego Padres', 'Texas Rangers', 
                'Seattle Mariners', 'Toronto Blue Jays',  'Los Angeles Angels',
                'Colorado Rockies',  'Arizona Diamondbacks', 'St. Louis Cardinals',
                'Washington Nationals', 'Tampa Bay Rays', 'Miami Marlins']



#Team colors-Section 2 of the dashboard updates plots to match current team colors. For non-current teams, the following deafult colors are used
DEFAULT_COLORS = ['#636EFA', '#BAB0AC', '#EF553B']
COLOR_DICT =  {'Cincinnati Reds':['#C6011F', '#BAB0AC', '#000000'],  'Pittsburgh Pirates':['#27251F', '#BAB0AC', '#FDB827'], 
                'Philadelphia Phillies':['#E81828', '#BAB0AC', '#002D72'], 'Chicago White Sox':['#27251F', '#FFFFFF', '#C4CED4'], 
                'Detroit Tigers':['#0C2340', '#BAB0AC', '#FA4616'], 'Baltimore Orioles':['#DF4601', '#BAB0AC', '#000000'], 
                'Milwaukee Brewers':['#12284B', '#BAB0AC', '#FFC52F' ], 'Chicago Cubs':['#0E3386', '#BAB0AC', '#CC3433'],  
                'Boston Red Sox':['#BD3039' ,  '#BAB0AC', '#0C2340'], 'New York Yankees':['#0C2340', '#E4002C', '#C4CED3'], 
                'Cleveland Indians':['#0C2340','#BAB0AC', '#E31937' ], 'San Francisco Giants':['#FD5A1E', '#EFD19F', '#27251F'], 
                'Los Angeles Dodgers':['#005A9C', '#A5ACAF', '#EF3E42'], 'Minnesota Twins':['#002B5C', '#B9975B', '#D31145'], 
                'New York Mets':['#002D72', '#BAB0AC', '#FF5910'],  'Houston Astros':['#002D62','#BAB0AC' ,'#EB6E1F'],  
                'Atlanta Braves':['#CE1141', '#EAAA00', '#13274F'], 'Oakland Athletics':['#003831', '#A2AAAD', '#EFB21E'], 
                'Kansas City Royals':['#004687', '#BAB0AC','#BD9B60'], 'San Diego Padres':['#2F241D','#BAB0AC', '#FFC425' ], 
                'Texas Rangers':['#003278','#BAB0AC', '#C0111F' ], 'Seattle Mariners':['#0C2C56', '#C4CED4' ,'#005C5C'], 
                'Toronto Blue Jays':['#134A8E', '#E8291C','#1D2D5C'],  'Los Angeles Angels':['#BA0021' ,'#C4CED4', '#003263'],
                'Colorado Rockies':['#33006F','#C4CED4', '#000000'],  'Arizona Diamondbacks':['#A71930', '#E3D4AD', '#000000'], 
                'St. Louis Cardinals':['#C41E3A', '#FEDB00', '#0C2340'], 'Washington Nationals':['#AB0003', '#BAB0AC', '#14225A' ], 
                'Tampa Bay Rays':['#092C5C', '#F5D130', '#8FBCE6'], 'Miami Marlins':['#00A3E0', '#000000', '#41748D']}


NO_POSTSEASON_LIST = [1900, 1901, 1902, 1903, 1904, 1994] #Years no postseasons happened

#Get teams lists
ALL_TEAMS = []
FEDERATION_TEAMS = []
for key in TEAMS_ID_DICT:
    ALL_TEAMS.append(key)
    if TEAMS_ID_DICT[key][0:3] == 'Fed':
        FEDERATION_TEAMS.append(key)

#sort teams lists
ALL_TEAMS.sort()
FEDERATION_TEAMS.sort()
CURRENT_TEAMS.sort()

#Define functions for loading in data. Note all data is cached with @st.cache above function definition

@st.cache
def load_data_1():
    """
    Loads YearlyResultsMaster.csv which contains the Season Results for all major league teams going back to 1900

    Returns
    ------
    Dataframe containing the FranID, Team, Year, Number of Games Played, Wins Losses, Number of Series Played, Series Wins,
    Series Losses, Series Ties, Win Percentage and Series Win Percentage for all major league teams separeted out by year going back to 1900
    """
    master_yearly_results= pd.read_csv('YearlyResultsMaster.csv')
    master_yearly_results['WinPercent'] = master_yearly_results['Wins']/ master_yearly_results['NumberOfGames']
    master_yearly_results['SeriesWinPercent'] = master_yearly_results['SeriesWins']/ master_yearly_results['NumberOfSeries']
    master_yearly_results['WinPercent'] = master_yearly_results['Wins']/ master_yearly_results['NumberOfGames']
    master_yearly_results['SeriesWinPercent'] = master_yearly_results['SeriesWins']/ master_yearly_results['NumberOfSeries']

    return master_yearly_results

@st.cache
def load_data_2():
    """
    Loads LeagueGameResults.csv which contains the individual game results for all major league baseball games played since 1900

    Returns
    ------
    Dataframe containing the date of each game, the home team name and FranhiseID for each team, the runs scored by each team, and the winner. 
    """
    working_df = pd.read_csv('LeagueGameResults.csv')
    working_df['Winner'] = np.where(working_df['Home_Team_Score'] > working_df['Away_Team_Score'] , working_df['Home_Team'], working_df['Away_Team'])
    working_df['Date'] = pd.to_datetime(working_df["Date"])
    working_df['Season'] = pd.DatetimeIndex(working_df['Date']).year
    return working_df


@st.cache
def load_data_3():
    """
    Loads 'PostSeasonStartDates.csv' which contains the year and postseason start date for each MLB season

    Returns
    ------
    Dataframe containing the year and postseason start date for each MLB season
    """
    post_season_marker_df = pd.read_csv('PostSeasonStartDates.csv')
    post_season_marker_df = post_season_marker_df.set_index('Season')
    return post_season_marker_df

@st.cache
def load_data_4():
    """
    Loads 'MasterYearlyResultsWithPlayoffs' which contains the year and postseason start date for each MLB season

    Returns
    ------
    Dataframe containing the compiled results plus postseason outcome results for visualization purposes
    """
    master_yearly_results_with_playoffs = pd.read_csv('MasterYearlyResultsWithPlayoffs.csv')
    return master_yearly_results_with_playoffs





#Load in Data
data_load_state = st.text('Loading data...')

master_yearly_results= load_data_1()
working_df = load_data_2()
post_season_marker_df = load_data_3()
master_yearly_results_with_playoffs = load_data_4()

data_load_state.text("")


# ------- Define Functions ----------------------------------------------------------------------------------


def get_team_and_years(team, start_year = None, end_year = None, historical_results = True, as_helper_function = False):

    """
    Parameters
    ----------
    team: string
        Team of interest
    start_year : integer, optional
        first year of data to pull. If none provided, the lowest year in team existence is set
    end_year :integer, optional
        last year of data to pull. If none provided, the lowest year in team existence is set
    historical_results : boolean, optional
        Whether or not to include all results from a franchise or just the results tied to the exact name selected. 
        For exmaple, include New York Highlanders Results for New York Yankees or not? Default is True.
    as_helper_function : boolean, optional
        Whether or not to return a plotly table or dataframe. Default is false which returns a plotly table,
        while true would return a dataframe


    Returns
    -------
    Either a dataframe or a plotly table of the results. 

    Columns of final dtaframe/table: Team, Year, Number of Games Played, Wins Losses, Number of Series Played, 
    Series Wins, Series Losses, Series Ties, Win Percentage and Series Win Percentage for each year of interest.
    """
    
    #If including historical results, we need to use the unique identifier for the franchise associated with the team (FranID)
    if historical_results:
        fran_id = TEAMS_ID_DICT[team]
    
        #TeamInQuotes= "'" + FranID + "'"
        query = f"FranID == '{fran_id}'"
        results = master_yearly_results.query(query)
        results = results.sort_values(by = 'Year')

    #If not including historical results, we can query for just the exact team name provided
    else:
        
        #TeamInQuotes= "'" + Team + "'"
        #query = "Team == " + TeamInQuotes
        query = f"Team == '{team}'"
        results = master_yearly_results.query(query)

    #If no start_year provided, default to minumum
    if start_year == None:
        start_year = results['Year'].iloc[0]
    
    #If no end_year provided, default to maximum
    if end_year == None:
        end_year = results['Year'].iloc[-1]


    #filter results for range provided by start_year and end_year
    results = results[results['Year'] >= start_year]
    results = results[results['Year'] <= end_year]

    
    results = results.reset_index()
    results = results.drop(columns = ['index', 'Unnamed: 0', 'FranID'])
    

    #Creat and add final row with compiled results
    total_row = [ 'Total:', '-', results.NumberOfGames.sum(),  results.Wins.sum(),  results.Losses.sum(),
                results.NumberOfSeries.sum(), results.SeriesWins.sum(),  results.SeriesLosses.sum(),
                results.SeriesTies.sum(), (results.Wins.sum()/results.NumberOfGames.sum()),
                (results.SeriesWins.sum()/results.NumberOfSeries.sum())]     
    results.loc['Total:'] = total_row


    #Round Win and Series win percentages for readability
    results['WinPercent'] = results['WinPercent'].round(decimals = 3)
    results['SeriesWinPercent'] = results['SeriesWinPercent'].round(decimals = 3) 
    
    
    
    #If not being used as a helper function, default is to build and return a table of results
    if as_helper_function == False:
        results = results.rename(columns = {'NumberOfSeries':'Number of Series', 'NumberOfGames':'Number of Games', 'SeriesWins': 'Series Wins', 'SeriesLosses':'Series Losses',
                            'SeriesTies': 'Series Ties', 'WinPercent':'Win Percent', 'SeriesWinPercent':'Series Win Percent' })

        table = go.Figure(data=[go.Table(
            header=dict(values=list(results.columns), 
                        font=dict(color='black')),
            cells=dict(values=[results['Team'], results['Year'], results['Number of Games'], results['Wins'],
                                results['Losses'], results['Number of Series'], results['Series Wins'], results['Series Losses'],
                                results['Series Ties'], results['Win Percent'], results['Series Win Percent']], 
                        font=dict(color='black')))

])
       
        table.update_layout(width=WIDE_WIDTH, height = TABLE_HEIGHT)
    
        return table
    
    #If being used as a helper function, return the dataframe itself
    else:
        return results

        

    


def get_team_and_years_plot(team, start_year = None, end_year = None, historical_results = True, as_helper_function = False):

    """
    Parameters
    ----------
    Team: string
        Team of interest
    start_year : integer, optional
        first year of data to pull. The default is the None, which will default to first 
        year of team existence when getTeamdAndYears is called
    end_year :integer, optional
        last year of data to pull. The default is the None, which will default to last year 
        of team existence when getTeamdAndYears is called

    Returns
    -------
    Plotly figure showing the teams win percentage and series win percantage for selected team over years of interest
    """

    #call get_team_and_years as helperfunction to get dataframe to draw data from
    results = get_team_and_years(team, start_year, end_year, as_helper_function=True)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results.Year, y=results.SeriesWinPercent,
            mode='lines',
            name='Series Win Percent'))
    fig.add_trace(go.Scatter(x=results.Year, y=results.WinPercent,
            mode='lines',
            name='Win Percent',
            line_color = '#2ca02c'))
    fig.add_hline(y=0.5, line_color = 'Red')
    fig.update_yaxes(range = [.2,.8], title="Win Percent") 
    fig.update_xaxes(title="Year") 
    fig.update_layout(width=WIDE_WIDTH, title = "Overall Win Percent and Series Win Percent by Year")
    return fig

    
def get_season_helper_function(team, year):

    """
    Parameters
    ----------
    team: string
        team of interest
    year : integer, optional
        year to pull data from

    Returns
    -------
    Dataframe for all games played by team of interest in year of interest including postseason
    Dataframe columns: Home Team, Home Team Score, Away Team, Away Team Score, Date, Winner, 
    Home Team FranID, Away Team FranID

    Used by
    -------
    get_one_year_regular_season
    get_one_year_playoffs
    """


    #Set up query string for team of interest
    #TeamInQuotes= "'" + Team + "'"
    #query1 = "Home_Team == " + TeamInQuotes + " | Away_Team == " + TeamInQuotes
    query_1 = f"Home_Team == '{team}'  | Away_Team == '{team}'"

    #query working_df for all games matching team of interest
    df_1 = working_df.query(query_1)
    df_1 = df_1.reset_index()

    #filter dataframe
    #query2 = "Season == " + str(Year)
    query_2 = f"Season == {str(year)}"
    df_1 = df_1.query(query_2)

    
    #return resulting dataframe
    return df_1


def get_one_year_results_full(team, year):

    
    """
    Parameters
    ----------
    team: string
        team of interest
    year : integer, optional
        year to pull data from

    Returns
    -------
    Plotly table of full season results (regular season and playoffs). 
    Dataframe columns: Home Team, Home Team Score, Away Team, Away Team Score, Date, Winner, 
    Team Winner (Did team of interest win), Cumulative Wins, Win Percentage to date

    """
    
    #Get regular season results
    df = get_one_year_regular_season(team, year, as_helper_function = True)

    #Get playoff results
    playoff_df = get_one_year_playoffs(team, year, as_helper_function = True)
    
    #If did not make playoffs add a row stating that   
    if len(playoff_df) == 0:
        final_row = ['-', '-', 'Did', 'Not', 'Make', 'Playoffs', '-', '-', '-']
        df.loc['-'] = final_row

    #If team did make playoffs add spacer row then playoff results
    else:
        spacer_row = ['Playoff', '-', '-', 'Results', '-', '-', '-', '-', '-']
        df.loc['-'] = spacer_row
        df = pd.concat([df, playoff_df], ignore_index=False)
    
    #create table
    df = df.rename(columns = {'Home_Team': 'Home Team', 'Home_Team_Score': 'Home Team Score', 'Away_Team': 'Away Team', 'WinPercent': 'Win Percent',
                            'Away_Team_Score': 'Away Team Score', 'Team_Winner': 'Result', 'Cumulative_Wins':'Cumulative Wins'})
    df= df.replace({'Result': {True: 'Win', False: 'Loss'}})

    table = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    font=dict(color='black')),

        cells=dict(values=[df['Home Team'], df['Home Team Score'], df['Away Team'], df['Away Team Score'],
                                df['Date'], df['Winner'], df['Result'], df['Cumulative Wins'], df['Win Percent']],
                    font=dict(color='black')))])

    #update table size (defaults declared at top)   
    table.update_layout(width=WIDE_WIDTH, height = TABLE_HEIGHT)
    
    #return table
    return table



def get_one_year_regular_season(team, year, as_helper_function = False):

    """
    Parameters
    ----------
    team: string
        team of interest
    year : integer
        year to pull data from

    as_helper_function : boolean, optional
        Whether or not to return a plotly table or dataframe. Default is false which returns a plotly table,
        while true would return a dataframe

    Returns
    -------
    Dataframe or plotly table for all games played by team of interest in year of interest during regular season. 
    Dataframe/Table Columns: Dataframe columns: Home Team, Home Team Score, Away Team, Away Team Score, Date, Winner, 
    Team Winner (Did team of interest win), Cumulative Wins, Win Percentage to date


    Called in
    -------
    get_one_year_results_full
    """
    
    #Call get_season_helper_function to get results for team and year of interest
    df = get_season_helper_function(team, year)
        
    #Check df is not empty
    if len(df) == 0:
        print('Invalid Year for Team Given. Please Try Again')
            
    else: 
        if year not in NO_POSTSEASON_LIST:
            post_season_check = post_season_marker_df.loc[year][0]
            playoff_df = df[df.Date >= post_season_check]
            df = df[df.Date < post_season_check]
        
     
        df = df.reset_index()    
        df.index = np.arange(1, len(df)+1)
    
        #Add columns for Team Winner (whether or not team of interest won), Cumulative Wins, and WinPercent (both to date)
        df['Team_Winner'] = df['Winner'] == team
        df['Cumulative_Wins'] = df['Team_Winner'].cumsum()
        df['WinPercent'] = df['Cumulative_Wins']/df.index

        #drop unnessary columns
        df = df.drop(columns = ['index', 'level_0', 'Home_FranID', 'Away_FranID'])
            
        #round off for readability
        df['WinPercent'] = df['WinPercent'].round(decimals = 3)
        df = df.drop(columns = 'Season')
            
        #convert to date
        df.Date = pd.DatetimeIndex(df.Date).strftime("%m-%d-%Y")

        #If not being used as a helper function, default is to build and return a table of results
        if as_helper_function == False:
            df = df.rename(columns = {'Home_Team': 'Home Team', 'Home_Team_Score': 'Home Team Score', 'Away_Team': 'Away Team', 'WinPercent': 'Win Percent',
                            'Away_Team_Score': 'Away Team Score', 'Team_Winner': 'Result', 'Cumulative_Wins':'Cumulative Wins'})
            df = df.replace({'Result': {True: 'Win', False: 'Loss'}})

            table = go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns),
                            font=dict(color='black')),

                    cells=dict(values=[df['Home Team'], df['Home Team Score'], df['Away Team'], df['Away Team Score'],
                                df['Date'], df['Winner'], df['Result'], df['Cumulative Wins'], df['Win Percent']],
                            font=dict(color='black')))])
       
            table.update_layout(width=WIDE_WIDTH, height = TABLE_HEIGHT)
            return table
        #if being used as a helper function, return a dataframe
        else: return df


def get_one_year_playoffs(team, year, as_helper_function = False):

    """
    Parameters
    ----------
    team: string
        team of interest
    year : integer
        year to pull data from

    as_helper_function : boolean, optional
        Whether or not to return a plotly table or dataframe. Default is false which returns a plotly table,
        while true would return a dataframe

    Returns
    -------
    Dataframe or plotly table for all games played by team of interest in year of interest during postseason. 
    Dataframe/Table Columns: Dataframe columns: Home Team, Home Team Score, Away Team, Away Team Score, Date, Winner, 
    Team Winner (Did team of interest win), Cumulative Wins, Win Percentage to date


    Called in
    -------
    get_one_year_results_full
    """
    
    #Call get_season_helper_function to get results for team and year of interest
    playoff_df = get_season_helper_function(team, year)
       
    #check valid playoff year
    if year not in NO_POSTSEASON_LIST:
        post_season_check = post_season_marker_df.loc[year][0]
        playoff_df = playoff_df[playoff_df.Date >= post_season_check]
    else: 
        post_season_check = '2099-01-01'
        playoff_df = playoff_df[playoff_df.Date >= post_season_check]
    
    #check df is not empty
    if len(playoff_df) == 0:
        if as_helper_function == True:
            return playoff_df
        else:
            return "Team Did Not Qualify for Playoffs in Year Provided"

    else:
        
        playoff_df = playoff_df.reset_index()    
        playoff_df.index = np.arange(1, len(playoff_df)+1)
    
        playoff_df['Team_Winner'] = playoff_df['Winner'] == team
        playoff_df['Cumulative_Wins'] = playoff_df['Team_Winner'].cumsum()
        playoff_df['WinPercent'] = playoff_df['Cumulative_Wins']/playoff_df.index


        playoff_df.Date = pd.DatetimeIndex(playoff_df.Date).strftime("%m-%d-%Y")
        
        playoff_df['WinPercent'] = playoff_df['WinPercent'].round(decimals = 3)
        playoff_df = playoff_df.drop(columns = ['index', 'level_0', 'Season', 'Home_FranID', 'Away_FranID'])
    
        if playoff_df.Team_Winner.iloc[-1] == True:
                
            final_row = ['-', '-', 'Won', 'World', 'Series', '!', '!', '-', '-']
        else: 
            final_row = ['-', 'Elimainated', 'in', 'Playoffs','by', playoff_df.Winner.iloc[-1], '-', '-', '-']
        playoff_df.loc['--'] = final_row
        
        #playoff_df.Date = pd.DatetimeIndex(playoff_df.Date).strftime("%m-%d-%Y")
        

        if as_helper_function == False:
            playoff_df = playoff_df.rename(columns = {'Home_Team': 'Home Team', 'Home_Team_Score': 'Home Team Score', 'Away_Team': 'Away Team', 'WinPercent': 'Win Percent',
                            'Away_Team_Score': 'Away Team Score', 'Team_Winner': 'Result', 'Cumulative_Wins':'Cumulative Wins'})
            playoff_df= playoff_df.replace({'Result': {True: 'Win', False: 'Loss'}})

            table = go.Figure(data=[go.Table(
                    header=dict(values=list(playoff_df.columns),
                            font=dict(color='black')),

                    cells=dict(values=[playoff_df['Home Team'], playoff_df['Home Team Score'], playoff_df['Away Team'], playoff_df['Away Team Score'],
                                playoff_df['Date'], playoff_df['Winner'], playoff_df['Result'], playoff_df['Cumulative Wins'], playoff_df['Win Percent']],
                            font=dict(color='black')))])
       
            table.update_layout(width=WIDE_WIDTH, height = TABLE_HEIGHT)
            return table
    
        else:
            return playoff_df


def made_playoffs(team, year):

    """
    Parameters
    ----------
    Team: string
        team of interest
    Year : integer
        year to pull data from


    Returns
    -------
    Boolean: True if the team played in the postseason for the year of interest, False otherwise
    """
    #Call get_one_year_playoffs to get playoff results for team/year of interest
    df = get_one_year_playoffs(team, year, as_helper_function = True)
    
    #If df not empty, team made postseason
    if len(df) != 0:
        return True
    else:
        return False


def won_world_series(team, year):

    """
    Parameters
    ----------
    Team: string
        team of interest
    Year : integer
        year to pull data from


    Returns
    -------
    Boolean: True if the team of interest won the World Series for the year of interest, False otherwise
    """
    

    #Call get_one_year_playoffs to get playoff results for team/year of interest
    df = get_one_year_playoffs(team, year, as_helper_function = True)
    
    #see if team has postseason results and if they won final game of result
    if len(df) == 0:
        return False
    elif team in FEDERATION_TEAMS:
        return False
    elif isinstance(df, str):
        return df
    else:
        if df.Team_Winner.iloc[-2] == True:
            return True
        else:
            return False



def get_record(team, year):
    """
    Parameters
    ----------
    team: string
        team of interest
    year : integer
        year to pull data from

    Returns
    -------
    A list contain information about the team for that season in the following format
    [Wins (integer), Losses (integer), WinPercent (Float), playoff_winner (string)]

    playoff_winner = who won the final series in that teams playoff result. Used to 
    determine who elimanted the team if they did not win the World Series. 

    """


    query = f"Team == '{team}'"

    results = master_yearly_results.query(query)
    results = results[results.Year == year]
    wins = results.Wins.iloc[0]
    losses = results.Losses.iloc[0]
    win_percent = round(results.WinPercent.iloc[0], 3)
    playoff_winner = np.NaN
    if made_playoffs(team, year):
        playoff_df = get_one_year_playoffs(team, year, as_helper_function=True)
        if playoff_df.Team_Winner.iloc[-2] != True:
            playoff_winner = playoff_df.Winner.iloc[-2]
    Record = [wins, losses, win_percent, playoff_winner]
    return Record


def get_bar_chart_1(team, year):
    """
    Parameters
    ----------
    team: string
        team of interest
    year : integer
        year of interest
    
    Returns
    -------
    Plotly bar chart with two bars- one showing win percent and loss percent stacked and 
    series win percent, series tie percent, and series loss percent stacked

    
    """

    df = get_team_and_years(team, year, year, as_helper_function = True)
    
    df['SeriesTiePercent'] = df.SeriesTies/df.NumberOfSeries
    df['SeriesLossPercent']= df.SeriesLosses/df.NumberOfSeries
    df['LossPercent'] = df.Losses/df.NumberOfGames
    
    df_2 = pd.DataFrame(
    dict(
        year=[year, year] * 3,
        layout=["Record", "Series Record"] * 3,
        response=["Win Percent", "Tie Percent", 'Loss Percent'] * 2,
        cnt=[df.WinPercent.iloc[-1], df.SeriesTiePercent.iloc[-1], df.LossPercent.iloc[-1],
        df.SeriesWinPercent.iloc[-1], 0, df.SeriesLossPercent.iloc[-1] ],
        response2=["Wins", "Ties", 'Losses'] * 2,
        cnt2=[df.Wins.iloc[-1], df.SeriesTies.iloc[-1], df.Losses.iloc[-1],
        df.SeriesWins.iloc[-1], 0, df.SeriesLosses.iloc[-1] ]
        ))  
      
    fig_1 = go.Figure()
    
    
    fig_1.update_layout(
    template="simple_white",
    xaxis=dict(title_text="Percent of Games Won/Lost (Left) and Percent of Series Won, Lost, and Tied (Right)"),
    yaxis=dict(title_text="Percent"),
    title = "Regular Season Results (Percentages)" ,
    barmode="stack",
    width = HALF_WIDTH,
    )
    

    if team in COLOR_DICT:
        colors = COLOR_DICT[team]
    else:
        colors = DEFAULT_COLORS

    for r, c in zip(df_2.response.unique(), colors):
        plot_df2 = df_2[df_2.response == r]
        fig_1.add_trace(
        go.Bar(x=[plot_df2.year, plot_df2.layout], y=plot_df2.cnt, name=r , marker_color=c) ,
            )
        

    return fig_1


def get_bar_chart_2(team, year):
    """
    Parameters
    ----------
    Team: string
        team of interest
    Year : integer
        year of interest
    
    Returns
    -------
    Plotly bar chart with two bars- one showing absolute number of wins and losses stacked and 
    one showing absolute number of series wins, series ties, and series losses stacked
    """
    df = get_team_and_years(team, year, year, as_helper_function = True)
    
    df['SeriesTiePercent'] = df.SeriesTies/df.NumberOfSeries
    df['SeriesLossPercent']= df.SeriesLosses/df.NumberOfSeries
    df['LossPercent'] = df.Losses/df.NumberOfGames
    
    df2 = pd.DataFrame(
    dict(
        year=[year, year] * 3,
        layout=["Record", "Series Record"] * 3,
        response=["Win Percent", "Tie Percent", 'Loss Percent'] * 2,
        cnt=[df.WinPercent.iloc[-1], df.SeriesTiePercent.iloc[-1], df.LossPercent.iloc[-1],
        df.SeriesWinPercent.iloc[-1], 0, df.SeriesLossPercent.iloc[-1] ],
        response2=["Wins", "Ties", 'Losses'] * 2,
        cnt2=[df.Wins.iloc[-1], df.SeriesTies.iloc[-1], df.Losses.iloc[-1],
        df.SeriesWins.iloc[-1], 0, df.SeriesLosses.iloc[-1] ]
        ))
    
    fig = go.Figure()
    
    fig.update_layout(
    template="simple_white",
    xaxis=dict(title_text="Number of Games Won/Lost (Left) and Number of Series Won, Lost, and Tied (Right)"),
    yaxis=dict(title_text="Count"),
    title = "Regular Season Results (Absolute Count)",
    barmode="stack",
    width = HALF_WIDTH,
    )
    
    if team in COLOR_DICT:
        colors = COLOR_DICT[team]
    else:
        colors = DEFAULT_COLORS
        
    for r, c in zip(df2.response2.unique(), colors):
        plot_df2 = df2[df2.response2 == r]
        fig.add_trace(
        go.Bar(x=[plot_df2.year, plot_df2.layout], y=plot_df2.cnt2, name=r , marker_color=c),
            )
    
    return fig


def get_one_year_plot(team, year):
    """
    Parameters
    ----------
    Team: string
        team of interest
    Year : integer
        year of interest
    
    Returns
    -------
    Plotly line chart with win and series win percent plotted against date in season.  
    """

    df = get_season_helper_function(team, year)
    
    df = df.reset_index()    
    df.index = np.arange(1, len(df)+1)        
    df['Team_Winner'] = df['Winner'] == team
    df['Cumulative_Wins'] = df['Team_Winner'].cumsum()
    df['WinPercent'] = df['Cumulative_Wins']/df.index
    df = df.drop(columns = ['index', 'level_0'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Date, y=df.WinPercent, mode='lines', name='Win Percent'))
    fig.add_hline(y=0.5, line_color = 'Red')
    fig.update_yaxes(range = [.1,.9], title = 'Win Percentage')
    fig.update_layout(width = WIDE_WIDTH, 
                      height = PLOT_HEIGHT, 
                      title = "Win Percentage by Date")
    return fig


def get_all_time_results_plot(teams, start_year, end_year):
    """
    Parameters
    ----------
    Team: string
        Team of interest
    start_year: integer, optional
        first year of data to pull. The default is the None, which will default to first 
        year of team existence when getTeamdAndYears is called
    end_year: integer, optional
        last year of data to pull. The default is the None, which will default to last year 
        of team existence when getTeamdAndYears is called

    Returns
    -------
    Plotly figure showing the win percentage vs series win percentage plotted for each team and year back to 1900
    """

    #filter by year
    filtered_results = master_yearly_results_with_playoffs[master_yearly_results_with_playoffs['Year'] > start_year]
    filtered_results = filtered_results[filtered_results['Year']<end_year]

    #filter by team
    filtered_results = filtered_results[filtered_results['Team'].isin(teams)]
    
    #differntiate by teams that made post season and were eliminated vs won world series (makes visual clearer)
    ws_index = filtered_results[filtered_results.WonWorldSeries == True]['MadePostSeason'].index
    filtered_results['MadePostSeason'] = filtered_results['MadePostSeason'].astype('string')
    for x in ws_index:
        filtered_results.at[x, 'MadePostSeason'] = 'True and Won World Series'

    
    #Round percentages for better presentation
    filtered_results["WinPercent"] = filtered_results["WinPercent"].round(decimals = 3)
    filtered_results["SeriesWinPercent"] = filtered_results["SeriesWinPercent"].round(decimals = 3)

    #rename columns for presentations
    filtered_results = filtered_results.rename(columns = {'MadePostSeason': 'Made Postseason?', 'WinPercent': 'Win Percent', 'SeriesWinPercent': 'Series Win Percent'})

    #select symbols and colors for visualization
    symbols = ['circle', 'diamond', 'star']
    color_sequence = ['#636EFA', '#EECA3B', '#EF553B']

    #Create Visual

    fig = px.scatter(filtered_results, x="Win Percent", y="Series Win Percent", color = "Made Postseason?", symbol = 'Made Postseason?',
                    width = 1200, height =850,
                    symbol_sequence = symbols,
                    color_discrete_sequence = color_sequence,
                    hover_data=["Team", "Year"])

    return fig


#######  ----------Streamlit Section----------------- ########

#title for whole page
st.title("Series-ly, You Have to Win Them")

#subheader for whole page
st.subheader("by [MacKenzye Leroy](https://mackenzye-leroy.com)")

##sidebar. Depending on choice, depends what is displayed
sidebar_selectbox = st.sidebar.radio(
    "",
    ("Home", "All-Time Results Visualized",
    "Season Over Season Results",
     "Single Season Results", "Top/Bottom 10 All-Time", 
     "Biggest Overachievers and Underachievers")
)





#### Home page:


if sidebar_selectbox == "Home":

    #Introduction to whole dashboard
    st.write("""Welcome! This dashboard is the result of a fairly simple question I wasn't able to find an answer to online--which MLB teams in history won
                the highest percentage of their regular season series (as opposed to games), and was a higher series win percentage indicative of playoff success? 
                When I realized there was no easy source of data to answer this question, I got to work making my own from other sources. I began by scraping the results of 
                all MLB games going back to 1900, cleaned the data up, and then calculated how many series each team played for each season
                and how many of those they won. If you're interested in my work collecting or cleaning the data, or my more rigorous statistical analysis of whether or
                not regular series win percentage was indicative of playoff success, check out my website [mackenzye-leroy.com](https://mackenzye-leroy.com), where I cover a lot of that work.
                If you're simply interested in playing around with some of the results, you're in the right place! Use the navigation bar on the left to navigate 
                to different widgets I built with the data. Each one is briefly described below.
                """)


    #Other dashboard otpions explained
    st.subheader('All-Time Results Visualized')

    st.write("""
            This widget plots the winning percentage and series winning percentage of all teams since 1900, as well as which teams made the playoffs and won 
            the World Series. You can filter by year, team, whether or not a team made the playoffs, and whether or not a team won the World Series. Click 
            "All-Time Results Visualized" in the navigation bar on the left to learn more!
            """)

    st.subheader('Season Over Season Results')

    st.write("""
            This widget allows you to look up the year-over-year results for any Major League team in baseball history from 1900 to 2021. 
            Click "Season Over Season Results" in the navigation bar on the left to learn more!
            """)

    st.subheader('Single Season Results')

    st.write("""
            This widget allows you to look up a single season of game results for any given team in Major League Baseball history.
            Click "Single Season Results" in the navigation bar on the left to learn more!
            """)

    st.subheader('Top/Bottom 10 All-Time')

    st.write("""
            This widget allows you to check out the best and worst teams in MLB history in terms of regular season win percentage. 
            Click "Top/Bottom 10 All-Time" in the navigation bar on the left to learn more!
            """)

    st.subheader('Biggest Overachievers and Underachievers')

    st.write("""
            This widget allows you to check out which teams have most overperformed and underperformed their win/loss record in terms of 
            series win percentage. Click "Biggest Overachievers and Underachievers" in the navigation bar on the left to learn more!
            """)



#### All Time Results Visualized:

elif sidebar_selectbox == "All-Time Results Visualized":

    #Option for selcting all teams, selecting all current teams (default), or clearing all teams
    
    teams_default = st.radio("Clear/Select Teams:", ('Select Current Teams', 'Select All Teams', 'Clear All Teams'))

    if teams_default == 'Select Current Teams':
        teams_default_result = CURRENT_TEAMS
    elif teams_default == 'Select All Teams':
        teams_default_result = ALL_TEAMS
    
    elif teams_default == 'Clear All Teams':
        teams_default_result = None

    #Team Select 
    team_input = st.multiselect("Teams", options = ALL_TEAMS, default = teams_default_result)

    #year select
    year_slider = st.slider("Years of Interest:", 1900, 2021, value=[1900, 2021])


    st.plotly_chart(get_all_time_results_plot(team_input, year_slider[0], year_slider[1]))

    #Explanation
    st.write("""
    The above graphic shows the regular season win percentage versus the regular season series win percentage of all MLB teams going back to 1900. 
    Teams represented as red diamonds made the postseason, and teams represented as gold stars won the World Series. Teams further to the right won a higher percentage of their 
    regular season games and teams higher up won a higher percentage of their regular season series. Obviously, those values are highly correlated (a team can't 
    win a lot of series without winning a lot of games), but there are still some interesting outliers. For example, the 1947 New York Yankees won over 63 percent
    of their games and ultimately won the World Series, but only about 47 percent of their regular season series. If you're interested in more outliers check out the "Biggest 
    Overachievers and Underachievers" tab on the left. We also see that while many World Series winning teams had both high win percentages and series 
    win percentages, plenty of teams with only slightly above average win percentages and series win percentages won the World Series. You can use the slider 
    above to filter years of interest and the box above to filter teams.
    \n
    
    Note: Results here include all teams since 1900 including some teams that have changed names, and some teams that no longer exist. If you're interested in learning 
    more about some of these historical teams, check out the "Season Over Season Results" tab on the left then the "Single Season Results" tab.
    """)

    


elif sidebar_selectbox == 'Season Over Season Results':

    #Title of Top Section
    st.subheader('Season Over Season Results')

    #Intro to top section
    st.write("""This widget allows you to look up the year-over-year results for any Major League team in baseball history within a selected range.
            A table with the results as well as a plot showing the team's win percentage and series win percentage during the span selected 
            will be automatically generated. You can also choose whether or not to include historical teams that may fall under this team's name. 
            For example, the New York Yankees are probably the most well-known team in baseball history. What many don't know though is that 
            they were originally founded as the Baltimore Orioles in 1901 and then changed names to the New York Highlanders from 1904 until
            1913 before ultimately landing on the world famous New York Yankees. Use the first dropdown to select whether or not you want to 
            include these historical results or not. You can also explore the results of the short-lived Federation League, 
            which existed from 1914-1915 by selecting "Federation League Teams." 
            """)

    #Team type dropdown
    team_dropdown_type = (st.selectbox(
    'Types of Teams', 
    ['Current Teams (historical results under other names included)', 
    'Current Teams (historical results under other names excluded)',
    'All Teams', 'Federation League Teams']))

    if team_dropdown_type == 'Current Teams (historical results under other names included)':
        team_dropdown = CURRENT_TEAMS
        historical_results_check = True

    elif team_dropdown_type ==  'Current Teams (historical results under other names excluded)':
        team_dropdown = CURRENT_TEAMS
        historical_results_check = False

    elif team_dropdown_type == 'Federation League Teams':
        team_dropdown = FEDERATION_TEAMS
        historical_results_check = True
    
    else: 
        team_dropdown = ALL_TEAMS
        historical_results_check = False

    #Select from given teams (updated when Team_dropdown_type is updated)
    team_option_1 = st.selectbox(
        'Select a Team:',
        team_dropdown)



    #include Historical results?
    if historical_results_check:
        franchise = TEAMS_ID_DICT[team_option_1]
        year_list_1 = master_yearly_results[master_yearly_results['FranID'] == franchise].Year
    else:
        year_list_1 = master_yearly_results[master_yearly_results['Team'] == team_option_1].Year


    #Year span selecter. Updates to reflect the max/min of whaichever team is selected above
    year_slider = st.slider("Years of Interest:", 1900, 2021, value=[min(year_list_1), max(year_list_1)])

    #only return plot if more than one year selected/available
    if year_slider[0] != year_slider[1]:

        st.plotly_chart(get_team_and_years_plot(team_option_1, start_year = year_slider[0], end_year=year_slider[1], historical_results = historical_results_check))

    #return table of reseults for years selected
    st.plotly_chart(get_team_and_years(team_option_1, start_year = year_slider[0], end_year=year_slider[1], historical_results = historical_results_check))

elif sidebar_selectbox == 'Single Season Results':

    #Single Season Results Section Section
    st.subheader('Single Season Results')

    #Introduction
    st.write("""This widget allows you to look up a single season of game results for any given team in Major League Baseball history. By default, the full regular season and playoff 
                schedule/results are returned as well as several plots. The first plot shows the win percentage of the team of interest over the course of the season. 
                The plot below on the left shows final win and loss percentage of the team in the given year as well as their final series win, loss, and tie percentage. 
                The final plot shows the final absolute count of wins and losses as well as series wins, ties, and losses for the season selected. If you are only interested 
                in playoff or regular season results, you can use the drop down below to select those. If you're only interested in the table of results or the plots, you
                can disable one or both of them with the checkboxes below. 
            """)

    #Type of Results
    results_type = st.selectbox(
        'Type Of Results',
        ['Full', 'Regular Season', 'Playoff'])


    #Include Plots? Default is True
    plots_2 = st.checkbox('Include plots?', value = True)

    #Include Table of results? Default is True
    results_2 = st.checkbox('Include game results?', value = True)

    #Select Team
    team_option_2 = st.selectbox(
        'Select a Team: ',
        ALL_TEAMS)

    #Find valid years for team selected
    year_list_2 = master_yearly_results[master_yearly_results['Team'] == team_option_2].Year
    year_list_2 = year_list_2.iloc[::-1]

    #Select Year. Only provides valid years for team selected above
    year_option = st.selectbox(
        'Select a Year:',
        year_list_2
        )

    #Check if team made playoffs for year selected
    playoffs = made_playoffs(team_option_2, year_option)

    if results_type == 'Playoff':
        #If team amde playoffs, allow user to look up just playoff results for year. 
        if playoffs:
            section_two_result = get_one_year_playoffs(team_option_2, year_option)
        else:
            section_two_result = None


    elif results_type == 'Regular Season':
        section_two_result = get_one_year_regular_season(team_option_2, year_option)
    else:
        section_two_result = get_one_year_results_full(team_option_2, year_option)

    #Get team Record list for year selected   
    record = get_record(team_option_2, year_option)



    #Construct results string bases on whether team made playoffs/won world series
    if playoffs:
        playoff_sentence = "qualified for the postseason"
        worldseries = won_world_series(team_option_2, year_option)
        if worldseries:
            worldseries_sentence = "and ultimately won the World Series!"
        else:
            worldseries_sentence =f"but were eliminated by the {record[3]}."
    else:
        playoff_sentence = 'did not qualify for the postseason.'
        worldseries_sentence = ''


    #Write result string
    info = f"In {year_option}, the {team_option_2} won {record[0]} games and lost {record[1]} for an overall win percentage of {record[2]}. They {playoff_sentence} {worldseries_sentence}"

    #Return result string
    st.subheader(info)

    #If plots box checked, generate win percentage plot and both bar charts
    if plots_2:

        st.plotly_chart(get_one_year_plot(team_option_2, year_option))

        col1, col2 = st.columns(2)

        col1.plotly_chart(get_bar_chart_1(team_option_2, year_option))

        col2.plotly_chart(get_bar_chart_2(team_option_2, year_option))

    #If results table box checked, return result (if valid)
    if results_2:
        if section_two_result != None:
            st.plotly_chart(section_two_result)


elif sidebar_selectbox == "Top/Bottom 10 All-Time":

    st.write("""Since this project started with the simple question of which teams in baseball won the highest percentage of their regular season series,
    it seems right to dedicate a section to those teams. The top table below shows the top 10 teams in MLB history in terms of series win percentage and 
    the bottom chart shows the bottom 10 (the teams with the worst series win percentages in MLB history). If you're interested in learning more about
    one or more of these incredible (or pitiful) seasons, be sure to make a note of the team name and year then head over to the "Singe Season Results" tab, 
    where you can look up those respective seasons and learn more.""")

    st.subheader('Top 10 All-Time')

    #get top 10 in Series Win Percentage, drop unneccessary columns, rename colunms for better presentation, and set index to 1-10
    top_10 = master_yearly_results_with_playoffs.sort_values(by = ['SeriesWinPercent'], ascending = False).head(10)
    top_10 = top_10.drop(columns = ['Unnamed: 0', 'LossPercent', 'SeriesLossPercent', 'SeriesTiePercent', 'LossDifference', 'FranID', 'Difference'])
    top_10 = top_10.rename(columns = {'NumberOfGames': 'Number Of Games', 'NumberOfSeries': 'Number Of Series', 'SeriesWins': 'Series Wins', 
                                                        'SeriesLosses': 'Series Losses', 'SeriesTies': 'Series Ties', 'WinPercent': 'Win Percent', 
                                                        'SeriesWinPercent': 'Series Win Percent', 'MadePostSeason': 'Made Postseason?', 'WonWorldSeries': 'Won World Series?'})
    top_10 = top_10.reset_index(drop=True)
    top_10.index = np.arange(1,len(top_10)+1)


    #return table
    st.table(top_10)

    st.subheader('Bottom 10 All-Time')

    #get bottom 10 in Series Win Percentage, drop unneccessary columns, rename colunms for better presentation, and set index to 1-10
    bottom_10 = master_yearly_results_with_playoffs.sort_values(by = ['SeriesWinPercent'], ascending = True).head(10)
    bottom_10 = bottom_10.drop(columns = ['Unnamed: 0', 'LossPercent', 'SeriesLossPercent', 'SeriesTiePercent', 'LossDifference', 'FranID', 'Difference'])
    bottom_10 = bottom_10.rename(columns = {'NumberOfGames': 'Number Of Games', 'NumberOfSeries': 'Number Of Series', 'SeriesWins': 'Series Wins', 
                                                        'SeriesLosses': 'Series Losses', 'SeriesTies': 'Series Ties', 'WinPercent': 'Win Percent', 
                                                        'SeriesWinPercent': 'Series Win Percent', 'MadePostSeason': 'Made Postseason?', 'WonWorldSeries': 'Won World Series?'})
    bottom_10 = bottom_10.reset_index(drop=True)
    bottom_10.index = np.arange(1,len(bottom_10)+1)

    #return table
    st.table(bottom_10)

elif sidebar_selectbox == 'Biggest Overachievers and Underachievers':

    st.write("""The following charts are all about outliers or teams that had unusually large differences in their  win percentages and their 
    series win percentages (regular season). I classified an overachiever as a team that won a higher percentage of their regular season series than their regular season games. 
    In all of MLB history, there have only been 172 such teams (out of over 2500 total teams!). Among that already select group, the top chart below shows the 10 teams with 
    the biggest differences in series win percentage and win percentage. I've crowned these teams as the biggest overachievers in MLB history. These teams were abnormally good at winning 
    series given their overall record. On the flip side, the bottom chart shows the 10 teams in MLB history with the largest difference in win percentage 
    and series win percenatge in the other direction--teams that won a much higher percentage of their games than their series. If you're interested in learning more about
    one or more of these unusual seasons, be sure to make a note of the team name and year then head over to the "Singe Season Results" tab, where you can look up those
    respective seasons and learn more.""")

    st.subheader('Biggest Overachievers')

    #get top 10 in terms of difference, drop unneccessary columns, rename colunms for better presentation, and set index to 1-10
    over_achievers = master_yearly_results_with_playoffs.sort_values(by = ['Difference'], ascending = False).head(10)
    over_achievers = over_achievers.drop(columns = ['Unnamed: 0', 'LossPercent', 'SeriesLossPercent', 'SeriesTiePercent', 'LossDifference', 'FranID'])
    over_achievers = over_achievers.rename(columns = {'NumberOfGames': 'Number Of Games', 'NumberOfSeries': 'Number Of Series', 'SeriesWins': 'Series Wins', 
                                                        'SeriesLosses': 'Series Losses', 'SeriesTies': 'Series Ties', 'WinPercent': 'Win Percent', 
                                                        'SeriesWinPercent': 'Series Win Percent', 'MadePostSeason': 'Made Postseason?', 'WonWorldSeries': 'Won World Series?'})
    over_achievers = over_achievers.reset_index(drop=True)
    over_achievers.index = np.arange(1,len(over_achievers)+1)
    #return table
    st.table(over_achievers)

    st.subheader('Biggest Underachievers')
    #get bottom 10 in terms of difference, drop unneccessary columns, rename colunms for better presentation, and set index to 1-10
    under_achievers = master_yearly_results_with_playoffs.sort_values(by = ['Difference'], ascending = True).head(10)
    under_achievers = under_achievers.drop(columns = ['Unnamed: 0', 'LossPercent', 'SeriesLossPercent', 'SeriesTiePercent', 'LossDifference', 'FranID'])
    under_achievers = under_achievers.rename(columns = {'NumberOfGames': 'Number Of Games', 'NumberOfSeries': 'Number Of Series', 'SeriesWins': 'Series Wins', 
                                                        'SeriesLosses': 'Series Losses', 'SeriesTies': 'Series Ties', 'WinPercent': 'Win Percent', 
                                                        'SeriesWinPercent': 'Series Win Percent', 'MadePostSeason': 'Made Postseason?', 'WonWorldSeries': 'Won World Series?'})
    under_achievers = under_achievers.reset_index(drop=True)
    under_achievers.index = np.arange(1,len(under_achievers)+1)
    #return table
    st.table(under_achievers)



#final contact note
st.write('')
st.write('Have a question? Reach out to me! You can find contact info on my website [mackenzye-leroy.com](https://mackenzye-leroy.com)')



