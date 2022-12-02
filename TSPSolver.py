#!/usr/bin/python3
import copy
import queue
from queue import PriorityQueue

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools





class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		myCities = self._scenario.getCities()
		numCities = len(myCities)

		startTime = time.time()

		initialDistances = [[math.inf]*numCities for i in range(numCities)]
		#fill array with costs from edge to edge
		for i in range(numCities):
			for j in range(numCities):
				initialDistances[i][j] = myCities[i].costTo(myCities[j])

		startCityIndex = 0
		totalCostAndPath = self.greedyHelper(initialDistances, myCities, startCityIndex)
		totalCost = totalCostAndPath[0]
		#keep being greedy until we get a solution or fail
		while totalCost == math.inf and startCityIndex < numCities:
			startCityIndex += 1
			totalCostAndPath = self.greedyHelper(initialDistances, myCities, startCityIndex)
			totalCost = totalCostAndPath[0]

		endTime = time.time()
		results = {}
		solut = TSPSolution(totalCostAndPath[1])
		results['cost'] = totalCost
		results['time'] = endTime - startTime
		results['count'] = startCityIndex # how many times we tried to get the best solution
		results['soln'] = solut
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	def greedyHelper(self, initialDistances, myCities, startCityIndex):
		visited = {}
		route = []
		numCities = len(myCities)
		for city in myCities:
			visited[city] = False

		currCity = myCities[startCityIndex]
		route.append(currCity)
		currCityIndex = startCityIndex
		# i had originally put original city is marked as being visited so the looping works, but we need to be sure to
		# add the cost from going to the last city back to the first
		# so even though we have a start city, we don't have a start visit.
		numCitiesVisited = 1
		visited[currCity] = True


		totalCost = 0

		# while we haven't visited each one, keep going
		# find cheapest nonvisited
		while numCitiesVisited != numCities:
			nextCityInd, cost = self.findCheapestNonvisited(initialDistances, visited, currCityIndex, numCities,
															myCities)
			#cost == math.inf means we got ourselves stuck. Too greedy....
			if cost == math.inf:
				return math.inf, None
			totalCost += cost
			currCity = myCities[nextCityInd]
			currCityIndex = nextCityInd
			visited[currCity] = True
			route.append(currCity)
			numCitiesVisited += 1

		#loop from the last to the first
		lastToFirstCost = initialDistances[currCityIndex][startCityIndex]
		if lastToFirstCost == math.inf:
			return math.inf, route
		else:
			return totalCost + lastToFirstCost, route


	def findCheapestNonvisited(self, initialDistances, visited, sourceCity, numCities, myCities):
		lowestNum = math.inf
		indexOfLowest = math.inf
		for i in range(numCities):
			if not visited[myCities[i]]:
				value = initialDistances[sourceCity][i]
				if value == 0:
					return i, 0
				if value < lowestNum:
					lowestNum = value
					indexOfLowest = i
		return indexOfLowest, lowestNum

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):

		myCities = self._scenario.getCities()
		numCities = len(myCities)

		startTime = time.time()
		greedyResponse = self.greedy()
		bssf = greedyResponse["soln"].cost
		#initialize array
		initialDistances = [[math.inf]*numCities for i in range(numCities)]
		workingDistances = [[math.inf] * numCities for i in range(numCities)]
		#fill array with costs from edge to edge
		for i in range(numCities):
			for j in range(numCities):
				initialDistances[i][j] = myCities[i].costTo(myCities[j])
				workingDistances[i][j] = myCities[i].costTo(myCities[j])


		bestCaseValue = self.getBestCaseScenario(workingDistances, numCities)
		if bestCaseValue == bssf:
			return greedyResponse

		pq = queue.PriorityQueue()

		#make children states
		#need a structure with:
		#path so far
		#cost so far
		#current matrix

		parentState = {"path": [0],
					  "bestCaseCost":bestCaseValue,
					  "matrix":copy.deepcopy(workingDistances)}

		#starting with first state
		totalStatesCreated = 1
		totalStatesPruned = 0
		totalTimesBSSFUpdate = 0
		maxSizeOfQueue = 1
		pq.put((parentState["bestCaseCost"], id(parentState), parentState))
		bestStateSoFar = parentState

		#generate children states


		while not pq.empty() and time.time() - startTime < time_allowance:
			#if we haven't tried travelling there
			parentState = pq.get()[2]
			#this should passively prune as needed.
			if parentState["bestCaseCost"] > bssf:
				totalStatesPruned += 1
				continue

			if len(parentState["path"]) == numCities:
				if parentState["bestCaseCost"] <= bssf:
					#this may need to be a deep copy
					bestStateSoFar = parentState
					bssf = bestStateSoFar["bestCaseCost"]
					totalTimesBSSFUpdate +=1
			else:
				#search state expansion
				for i in range(numCities):
					if i not in parentState["path"]:
						#create a child state
						newChild = self.createChild(parentState, i, numCities)
						totalStatesCreated += 1
						if newChild["bestCaseCost"] <= bssf:
							#this priorityQueueKey determination is arbitrary and can be tweaked. Right now we emphasize
							#the ones closer to a solution
							#This heuristic seems great
							priorityQueueKey = newChild["bestCaseCost"] / (len(newChild["path"]) / 3)
							#but commented out to see how great
							#priorityQueueKey = newChild["bestCaseCost"]
							# have to do id of new child because if priorityQueueKey equals another one, it compares
							# the next thing in the tuple
							pq.put((priorityQueueKey, id(newChild), newChild))
							if ((pq.qsize()) > maxSizeOfQueue):
								maxSizeOfQueue = pq.qsize()
						else:
							totalStatesPruned += 1

		route = []
		numericPath = bestStateSoFar["path"]
		if len(numericPath) < numCities:
			greedyResponse["count"] = 0
			return greedyResponse
		for i in range(numCities):
			route.append(myCities[numericPath[i]])

		endTime = time.time()
		solut = TSPSolution(route)
		results = {}
		results['cost'] = bssf
		results['time'] = endTime - startTime
		results['count'] = totalTimesBSSFUpdate # how many times we tried to get the best solution
		results['soln'] = solut
		results['max'] = maxSizeOfQueue
		results['total'] = totalStatesCreated
		results['pruned'] = totalStatesPruned
		return results

	def createChild(self, parentState, destCityIndex, numCities):
		currCityIndex = parentState["path"][-1]
		newBestCaseCost = parentState["bestCaseCost"] + parentState["matrix"][currCityIndex][destCityIndex]
		newMatrix = copy.deepcopy(parentState["matrix"])

		#infinity out those that can't be visited now
		for i in range(numCities):
			newMatrix[currCityIndex][i] = math.inf
			newMatrix[i][destCityIndex] = math.inf

		newMatrix[destCityIndex][currCityIndex] = math.inf

		#get new cost matrix
		costToReduce = self.getBestCaseScenario(newMatrix, numCities)

		newBestCaseCost = newBestCaseCost + costToReduce
		newPath = copy.deepcopy(parentState["path"])
		newPath.append(destCityIndex)
		childState = {"path": newPath,
					  "bestCaseCost":newBestCaseCost,
					  "matrix":copy.deepcopy(newMatrix)}
		return childState



	def getBestCaseScenario(self, initialDistances, numCities):
		totalCost = 0


		#Rows
		for i in range(numCities):
			lowestNumber = self.findLowestNumberInRow(initialDistances, i, numCities)
			if lowestNumber != math.inf:
				totalCost+=lowestNumber
			if lowestNumber != 0 and lowestNumber != math.inf:
				for j in range(numCities):
					initialDistances[i][j] -= lowestNumber

		#Cols
		for j in range(numCities):
			lowestNumber = self.findLowestNumberInColumn(initialDistances, j, numCities)
			if lowestNumber != math.inf:
				totalCost += lowestNumber
			if lowestNumber != 0 and lowestNumber != math.inf:
				for i in range(numCities):
					initialDistances[i][j] -= lowestNumber


		return totalCost



	def findLowestNumberInRow(self, allCityDistances, rowNum, numCities):
		lowestNum = math.inf

		for i in range(numCities):
			value = allCityDistances[rowNum][i]
			if value == 0:
				return 0
			if value < lowestNum:
				lowestNum = value
		return lowestNum

	def findLowestNumberInColumn(self, allCityDistances, colNum, numCities):
		lowestNum = math.inf
		for i in range(numCities):
			value = allCityDistances[i][colNum]
			if value == 0:
				return 0
			if value < lowestNum:
				lowestNum = value
		return lowestNum


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass







