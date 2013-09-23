# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.spatial.distance import pdist,squareform
from matplotlib.path import Path
from matplotlib import animation

from util import distance as distance_lat_long

import argparse

def read_cities(file_name):
	names = []
	city_x = []
	city_y = []

	scale = 1

	with open(file_name, 'r') as f:
		reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
		for row in reader:
			names.append(row[0])
			city_x.append(float(row[1]) * scale)
			city_y.append(float(row[2]) * scale)
	return names, city_x, city_y

def read_germany(file_name):
	ger_x = []
	ger_y = []

	scale = 1

	with open(file_name, 'r') as f:
		reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
		for row in reader:
			ger_x.append(float(row[0]) * scale)
			ger_y.append(float(row[1]) * scale)
	return ger_x, ger_y

def genetic_salesman(city_x, city_y, iterations, crossing, mutation):
	assert len(city_x) == len(city_y)

	#Number of cities
	cities = len(city_x)

	np.random.seed(seed=13374223)
	locations=np.zeros((cities,2)) #Zufällige Festlegung der Orte

	for i, (xp, yp) in enumerate(zip(city_x, city_y)):
		locations[i,:]=[xp,yp]					
	
	# Calculates distance between all cities
	distances = squareform(pdist(locations, distance_lat_long))

	np.random.seed()

	bestDist=np.zeros(iterations) #In diesem Array wird für jede Iteration die beste Distanz gespeichert
	#Erzeugen einer zufälligen Startpopulation
	population=np.zeros((cities,cities+1))
	for j in range(cities):
			population[j,0:cities]=np.random.permutation(cities)
			population[j,cities]=population[j,0]

	cost=np.zeros(cities)#Speichert die Kosten jedes Chromosoms der aktuellen Population
	#Berechnung der Kosten jedes Chromosoms

	x_paths = np.zeros((iterations, cities + 1))
	y_paths = np.zeros((iterations, cities + 1))

	##################################################################################################
	for it in range(iterations):

			#1.Berechne Fitness der aktuellen Chromosomen#################################################
			for j,pop in enumerate(population):
					cost[j]=0
					for z in range(cities):
							cost[j]=cost[j]+distances[pop[z],pop[z+1]]

			sortedIndex=cost.argsort(axis=0)#Indizees der nach ansteigenden Kosten sortierten Chromosomen
			sortedCost=cost[sortedIndex] #die ansteigend sortierten Kosten
			bestDist[it]=sortedCost[0]
			sortedPopulation=population[sortedIndex] #Sortierung der Population nach ansteigenden Kosten
			InvertedCost=1/sortedCost #Berechung des Nutzen (Fitness) aus den Kosten
			#InvertedCost enthält die berechneten Fitness-Werte

			#2.Selektion: Zufällige Auswahl von Chromosomen aus der Population####################
			#Mit dem folgenden Prozess wird gewährleistet, dass die Wahrscheinlichkeit für die
			#Selektion eines Chromosoms umso größer ist, je größer sein Nutzenwert ist.
			InvertedCostSum = InvertedCost.sum()
			rn1=InvertedCostSum*np.random.rand()
			found1 = False
			index=1
			while not found1:
					if rn1<InvertedCost[:index].sum(axis=0):
							found1=index
					else:
							index+=1
			found1=found1-1
			equal=True
			while equal:
					rn2=InvertedCostSum*np.random.rand()
					found2 = False
					index=1
					while not found2:
							if rn2<InvertedCost[:index].sum(axis=0):
									found2=index
							else:
									index+=1
					found2=found2-1
					if found2 != found1:
							equal=False
			parent1=sortedPopulation[found1]
			parent2=sortedPopulation[found2]
			########## parent1 und parent2 sind die selektierten Chromsomen##############################



			#3.Kreuzung####################################################################################
			crossrn=np.random.rand()
			if crossrn<crossing:
					cp=np.ceil(np.random.rand()*cities)
					head1=parent1[:cp]
					tailind=0
					tail1=np.zeros(cities-cp+1)
					for a in range(cities):
							if parent2[a] not in head1:
									tail1[tailind]=parent2[a]
									tailind+=1
					tail1[-1]=head1[0]
					head2=parent2[:cp]
					tailind=0
					tail2=np.zeros(cities-cp+1)
					for a in range(cities):
							if parent1[a] not in head2:
									tail2[tailind]=parent1[a]
									tailind+=1
					tail2[-1]=head2[0]
					child1=np.append(head1,tail1)
					child2=np.append(head2,tail2)
			#child1 und child2 sind die Ergebnisse der Kreuzung###############################################


			#4. Mutation#########################################################################################
			mutrn=np.random.rand()
			if mutrn<mutation:
					mutInd=np.ceil(np.random.rand(2)*(cities-1))
					first=child1[mutInd[0]]
					second=child1[mutInd[1]]
					child1[mutInd[0]]=second
					child1[mutInd[1]]=first
					child1[-1]=child1[0]

			mutrn=np.random.rand()
			if mutrn<mutation:
					mutInd=np.ceil(np.random.rand(2)*(cities-1))
					first=child2[mutInd[0]]
					second=child2[mutInd[1]]
					child2[mutInd[0]]=second
					child2[mutInd[1]]=first
					child2[-1]=child2[0]
			#child1 und child2 sind die Resultate der Mutation################################################



			#5. Ersetze die bisher schlechtesten Chromosomen durch die neu gebildeten Chromosomen, falls die neuen
			#besser sind
			costChild1=0
			costChild2=0
			for z in range(cities):
					costChild1=costChild1+distances[child1[z],child1[z+1]]
					costChild2=costChild2+distances[child2[z],child2[z+1]]
			replace1=False
			replace2=False
			index=cities-1
			while index > 0:
					if sortedCost[index]>costChild1 and not replace1:
							if not np.ndarray.any(np.ndarray.all(child1==sortedPopulation,axis=1)):
									sortedPopulation[index]=child1
							replace1=True
					elif sortedCost[index]>costChild2 and not replace2:
							if not np.ndarray.any(np.ndarray.all(child2==sortedPopulation,axis=1)):
									sortedPopulation[index]=child2
							replace2=True
					if replace1 and replace2:
							break
					index=index-1
			population=sortedPopulation
			######################################Ende der Iteration#############################

			for i in range(cities+1):
				x_paths[it, i] = locations[sortedPopulation[0,i],0]
				y_paths[it, i] = locations[sortedPopulation[0,i],1]

	return x_paths, y_paths, bestDist

def plot_time(path_x, path_y, city_x, city_y, ger_x, ger_y, distances, names):
	for i in range(iterations):

		path_x, path_y = x_paths[i], y_paths[i]
		distance = distances[:i]

		"Plot"

		plt.figure(1)
		plt.subplot(121)
		#plt.plot(city_x,city_y)
		plt.grid(True)
		plt.hold(True)		

		plt.axis([6., 15., 47., 55.])
		ax = plt.gca()
		ax.set_autoscale_on(False)

		plt.plot(path_x, path_y,'r-')
		plt.plot(ger_x, ger_y, 'b,')

		for name, xp, yp in zip(names, city_x, city_y):
			plt.plot([xp],[yp],'ro')
			plt.text(xp+xshift,yp+yshift, name)

		plt.subplot(122)
		plt.grid(True)
		plt.plot(range(i),distance)

		plt.show()

def movietime(x_paths, y_paths, ger_x, ger_y, distances, names, filename='genetic_salesman.mp4'):
	"""
		Creates an animation of the traveling salesman steps
		by drawing one integration step after another
		and saves it.
		x:  Array of x coords
		y:  Array of y coords
		filename: File to save the animation to
	"""

	MOVIEWRITER = 'mencoder'

	# Limit plot to lat long of Germany
	XLIM = (6., 15.)
	YLIM = (47., 55.)
	REFRESHRATE = 10

	xshift=0.2
	yshift=0.2

	fig = matplotlib.pyplot.figure()

	# Map part

	ax1 = fig.add_subplot(121, autoscale_on=False, xlim=XLIM, ylim=YLIM)
	ax1.grid()
	line1, = ax1.plot([], [], 'o-', lw=2)

	# Plot borders 
	border, = ax1.plot([], [], 'b,', lw=1)

	# Plot names
	for name, xp, yp in zip(names, city_x, city_y):
		ax1.plot([xp],[yp],'ro')
		ax1.text(xp+xshift,yp+yshift, name)

	time_template = 'Iteration = %d'
	time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

	# Distance part

	I_LIM = (0., max(distances) * 1.1)

	ax2 = fig.add_subplot(122, autoscale_on=False, xlim=(0, len(distances)), ylim=I_LIM)
	ax2.grid()
	line2, = ax2.plot([], [], 'r-,', lw=2, label="Iterations", markersize=2)

	def init():
		line1.set_data([], [])
		line2.set_data([], [])
		time_text.set_text('')
		border.set_data(ger_x, ger_y)

		return line1, time_text, border, line2

	def animate(i):
		x = x_paths[i]
		y = y_paths[i]

		iteration_range = np.arange(0, i)
		
		distance = distances[:i]

		# Map part
		line1.set_data(x, y)
		time_text.set_text(time_template % i)

		# Distance part
		line2.set_data(iteration_range, distance)

		return line1, time_text, border, line2

	ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(distances), interval=REFRESHRATE, blit=True, init_func=init)
	#ani.save(filename, writer=animation.FFMpegFileWriter(), fps=30)

	plt.show()


if __name__ == '__main__':
	import os, sys
	os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--cities", default="cities.csv", help="CSV file with cities")
	parser.add_argument("-b", "--borders", default="german_borders.csv", help="CSV file with country borders")
	parser.add_argument("-i", "--iterations", type=int, default=5000, help="Number of generations/iterations")
	parser.add_argument("-c", "--crossing", type=float, default=0.99, help="Gene crossing probaility")
	parser.add_argument("-m", "--mutation", type=float, default=0.1, help="Mutation probability")

	args = parser.parse_args()
	
	#Definition von Konstanten für die Anzeige der Stadtindizess

	names, city_y, city_x = read_cities(args.cities)
	ger_x, ger_y = read_germany(args.borders)

	x_paths, y_paths, distances = genetic_salesman(city_x, city_y, args.iterations, args.crossing, args.mutation)

	#plot_time(x_paths, y_paths, city_x, city_y, distances, names)
	movietime(x_paths, y_paths, ger_x, ger_y, distances, names)

