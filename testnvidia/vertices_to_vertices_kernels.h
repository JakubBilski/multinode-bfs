#pragma once
#include "defines.h"

__global__
void NSquaredKernel(int noVertices, int iteration, int* dist, int* cAdjacencyList, int* rAdjacencyList, int* isVertexFrontierEmpty)
{
	if (blockIdx.x * blockDim.x + threadIdx.x < noVertices && dist[blockIdx.x * blockDim.x + threadIdx.x] == iteration)
	{
		for (int i = rAdjacencyList[blockIdx.x * blockDim.x + threadIdx.x]; i < rAdjacencyList[blockIdx.x * blockDim.x + threadIdx.x + 1]; i++)
		{
			if (dist[cAdjacencyList[i]] == INF)
			{
				dist[cAdjacencyList[i]] = iteration + 1;
			}
		}
		isVertexFrontierEmpty[0] = 0;
	}
}

__global__
void serialGatheringAtomicArrayKernel(int noVertices, int iteration, int* dist, int* cAdjacencyList, int* rAdjacencyList, int* inQueue, int* outQueue, int* outCounter)
{
	if (blockIdx.x * blockDim.x + threadIdx.x < noVertices)
	{
		int vertex = inQueue[blockIdx.x * blockDim.x + threadIdx.x];
		for (int offset = rAdjacencyList[vertex]; offset < rAdjacencyList[vertex + 1]; offset++)
		{
			__syncthreads();
			int j = cAdjacencyList[offset];
			if (dist[j] == INF)
			{
				dist[j] = iteration + 1;
				int queueIndex = atomicAdd(outCounter, 1);
				outQueue[queueIndex] = j;
			}
		}
	}
}

__global__
void warpBasedGatheringAtomicArrayKernel(int noVertices, int iteration, int* dist, int* cAdjacencyList, int* rAdjacencyList, int* inQueue, int* outQueue, int* outCounter)
{
	__shared__ volatile int command[NO_THREADS / 32][3];
	int warpId = threadIdx.x / 32;
	if (warpId * 32 < noVertices)
	{
		int offset, offsetEnd;
		if (blockIdx.x * blockDim.x + threadIdx.x < noVertices)
		{
			int vertex = inQueue[blockIdx.x * blockDim.x + threadIdx.x];
			offset = rAdjacencyList[vertex];
			offsetEnd = rAdjacencyList[vertex + 1];
		}
		else
		{
			offset = 0;
			offsetEnd = 0;
		}
		while (1)
		{
			command[warpId][0] = -1;
			if (offsetEnd != 0)
			{
				command[warpId][0] = threadIdx.x;
			}
			if (command[warpId][0] == -1)
			{
				break;
			}
			if (command[warpId][0] == threadIdx.x)
			{
				command[warpId][1] = offset;
				command[warpId][2] = offsetEnd;
				offsetEnd = 0;
			}
			volatile int index = command[warpId][1] + threadIdx.x % 32;
			int adjEnd = command[warpId][2];
			while (index < adjEnd)
			{
				volatile int j = cAdjacencyList[index];
				if (dist[j] == INF)
				{
					dist[j] = iteration + 1;
					volatile int queueIndex = atomicAdd(outCounter, 1);
					outQueue[queueIndex] = j;
				}
				index += 32;
			}
		}
	}
}