#pragma once
#include <list>
#include "defines.h"

void CPUSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int selection)
{
	std::list<int> vertexQueue;
	std::list<int> newVertexQueue;
	int* dist = (int*)malloc(sizeof(int)*noVertices);
	for (int i = 0; i < noVertices; i++)
	{
		dist[i] = INF;
	}
	dist[startingVertex] = 0;
	vertexQueue.push_back(startingVertex);
	int iteration = 0;
	while (!vertexQueue.empty())
	{
		for (auto vertex : vertexQueue)
		{
			for (int i = rAdjacencyList[vertex]; i < rAdjacencyList[vertex+1]; i++)
			{
				if (dist[cAdjacencyList[i]] == INF)
				{
					dist[cAdjacencyList[i]] = iteration + 1;
					newVertexQueue.push_back(cAdjacencyList[i]);
				}
			}
		}
		vertexQueue.clear();
		vertexQueue.swap(newVertexQueue);
		iteration++;
	}
}