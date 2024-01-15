import pandas as pd
import heapq
from collections import defaultdict
import datetime
from datetime import datetime, date, timedelta

# Function to parse time strings
def parse_time(time_str):
    #if time_str == '00:00:00':
     #   return None
    return datetime.strptime(time_str.strip("'"), '%H:%M:%S').time()

# Function to create a revised graph from the schedule DataFrame
def create_revised_graph(schedule_df):
    graph = defaultdict(list)
    grouped = schedule_df.groupby('Train No.')

    for train_no, group in grouped:
        sorted_stops = group.sort_values('islno')
        for i in range(len(sorted_stops) - 1):
            current_stop = sorted_stops.iloc[i]
            next_stop = sorted_stops.iloc[i + 1]

            edge = {
                'next_station': next_stop['station Code'].strip(),
                'train_no': train_no.strip(),
                'islno':current_stop['islno'] ,
                'departure_time': parse_time(current_stop['Departure time']),
                'arrival_time':parse_time(next_stop['Arrival time']),
                'distance': next_stop['Distance'] - current_stop['Distance']
            }
            graph[current_stop['station Code'].strip()].append(edge)

    return graph
 
 

# Cost function for distance
def revised_distance_cost(edge):
    return edge['distance']

# Cost function for the number of stops
def stops_cost(edge):
    return 1  # Each edge traversal is a single stop

# Cost function for the ticket price
def ticket_price_cost(previous_edge, current_edge):
    
    if previous_edge is None or previous_edge['train_no'] != current_edge['train_no']:
        return 1  # New train, so a new ticket is needed

    #elif   current_edge['departure_time'] > current_edge['arrival_time']:
    #  return 1# midnight is passed, so a new ticket is needed
    
    return 0  # Same train, no new ticket needed



def calculate_time_difference(start_time, end_time):
    """Calculate the time difference, considering overnight journeys."""
    # Create datetime objects for the same date
    base_date = date(2000, 1, 1)  # Arbitrary date
    datetime_start = datetime.combine(base_date, start_time)
    datetime_end = datetime.combine(base_date, end_time)

    # Adjust for overnight journey
    if datetime_start > datetime_end:
        datetime_end += timedelta(days=1)

    # Calculate the difference in seconds and convert to hours
    time_difference = (datetime_end - datetime_start).seconds / 3600.0
    return time_difference



def arrival_time_cost(previous_edge, current_edge, current_time):
    """Calculate the cost in terms of time taken including waiting for the next train."""
    if previous_edge is None:
        # First edge
        w=datetime.strptime(current_time, '%H:%M:%S').time()
        return calculate_time_difference(w, current_edge['arrival_time'])

    # Calculate waiting time at the station for the next train
    waiting_time = calculate_time_difference(previous_edge['arrival_time'], current_edge['departure_time'])
    journey_time = calculate_time_difference(current_edge['departure_time'], current_edge['arrival_time'])
    

    return waiting_time + journey_time

def arrival_time_cost2(previous_edge, current_edge, current_time):
    """Calculate the cost in terms of time taken including waiting for the next train."""
    if previous_edge is None:
        # First edge
        w=datetime.strptime(current_time, '%H:%M:%S').time()
        if w > current_edge['departure_time']:
            return 1
    '''   
    if previous_edge != None and previous_edge['departure_time'] > current_edge['arrival_time']:
            return 1'''
    return 0


def timedelta_to_str(delta):
     

    # Calculate total seconds
    total_seconds = int(delta.seconds)

    # Calculate hours, minutes, and seconds
    days=delta.days  
    hours =  total_seconds  // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Format the string
    formatted_time = f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"

    return formatted_time
def solution_Connection(path):
    connection=''
    totalcost=0
    prev_train_no=0
    prev_islno=0
    prev_station=''
    for station,train_no, islno, cost in path:  
        train_no=train_no.strip("'")
        if prev_train_no==0:
            prev_train_no=train_no
            prev_islno=islno
            prev_station=station
            connection=f"{train_no} : {islno}"

        if prev_train_no!=train_no:
            connection=f"{connection} -> {prev_islno} ; {train_no} : {islno-1}"

        #connection=f"{connection} -> {islno}"
        #totalcost+=cost
        totalcost=cost

        prev_train_no=train_no
        prev_islno=islno
        prev_station=station

    connection=f"{connection} -> {prev_islno}"
    return connection,totalcost

def dijkstra(graph, start_station, end_station, start_time, cost_func,cost_func_type):
    queue = [(0,0,0, start_station, None, start_time,0, [])]  # (cumulative time cost, train_no, islno, station, previous edge, current time, price unique id, path taken)
    visited = set()
    unique_id=0
     
    while queue:
        cost,train_no,islno, station, previous_edge, current_time, _, path = heapq.heappop(queue)

        if station in visited:
            continue
        visited.add(station)
         

        if station == end_station:
            return path + [(station,train_no,islno+1, cost)] ,current_time
            

        for edge in graph[station]:
            next_station = edge['next_station']
            
            if next_station not in visited :
                tno=edge['train_no']
                ino=edge['islno']
                
                if edge['train_no']!=train_no and train_no!=0:
                    tno=train_no
                    ino=islno+1
                    

                new_current_time=''
                next_cost=0
                

                if cost_func_type=='stops':
                    next_cost = cost + cost_func(edge)
                    
                elif cost_func_type=='distance':
                    next_cost = cost + cost_func(edge)
                elif cost_func_type=='price':
                    
                    next_cost = cost + cost_func(previous_edge, edge)
                    unique_id += 1  # Increment the unique identifier
                    #heapq.heappush(queue, (next_cost, unique_id,edge['train_no'], next_station, edge, path + [(station, cost)]))
                elif cost_func_type=='arrivaltime':

                    additional_time = cost_func(previous_edge, edge, current_time)                   
                    next_cost = cost + additional_time                   
                    #new_current_time = (datetime.strptime(current_time, '%H:%M:%S') + timedelta(hours=additional_time)).strftime('%H:%M:%S')
                    new_current_time=edge['arrival_time'].strftime('%H:%M:%S')
                     
                    
                heapq.heappush(queue, (next_cost,edge['train_no'],edge['islno'], next_station, edge, new_current_time,unique_id, path + [(station,tno,ino, cost)]))

    return None

def main():

    # Load the schedule data
    schedule_path = 'schedule.csv' 
    schedule_df = pd.read_csv(schedule_path)
    # Create the revised graph
    revised_railway_graph = create_revised_graph(schedule_df)
    
    minischedule_path = 'mini-schedule.csv'  
    minischedule_df = pd.read_csv(minischedule_path)
    # Create the revised graph
    revised_railway_graphmini = create_revised_graph(minischedule_df)


    problems = pd.read_csv("problems.csv")
    solutions = []

    for index, problem in problems.iterrows():
       if True:#problem['CostFunction']=='price':

        problemNo=problem['ProblemNo']
        start_station = problem['FromStation']
        end_station = problem['ToStation']
        cost_function = problem['CostFunction']
        schedule_path=problem['Schedule']

        if schedule_path=="mini-schedule.csv":
            revised_graph = revised_railway_graphmini
        else:
            revised_graph=revised_railway_graph
        
        connection=''
        totalcost=0
        if cost_function=='distance':
            # Find the path with the shortest distance
            shortest_path_revised,_ = dijkstra(revised_graph, start_station, end_station, None, revised_distance_cost,'distance') 
            connection,totalcost=solution_Connection(shortest_path_revised)
             
        elif cost_function=='stops':
            # Find the path with the least number of stops
            path_least_stops,_=dijkstra(revised_graph, start_station, end_station, None, stops_cost,'stops')
            connection,totalcost=solution_Connection(path_least_stops)
             
        elif cost_function=='price':
            # Find the path with the minimum ticket price
            path_min_ticket_price,_=dijkstra(revised_graph, start_station, end_station, None, ticket_price_cost,'price')
            connection,totalcost=solution_Connection(path_min_ticket_price)
            
        else:
            start_time=cost_function.strip("arrivaltime ")
            # Find the path with the earliest arrival time
            path_earliest_arrival,final_time= dijkstra(revised_graph, start_station, end_station, start_time, arrival_time_cost,'arrivaltime')
            connection,cost=solution_Connection(path_earliest_arrival)
            
            days=  timedelta(hours=cost).days  
            if True:
                t1=parse_time(start_time) 
                t2=parse_time(final_time)
                if t1 > t2:
                    days=days+1

            totalcost=f"{days:02}:{final_time}"

        
    

        solutions.append({
            'ProblemNo': problemNo,
            'Connection': connection,
            'Cost': totalcost
        })

    #pd.DataFrame(solutions)
    df = pd.DataFrame(solutions)

    # Write the DataFrame to a CSV file
    df.to_csv('solutions.csv', index=False)
  
    print(solutions)


if __name__ == "__main__":
    main()