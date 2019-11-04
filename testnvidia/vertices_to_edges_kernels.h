#pragma once
#include "defines.h"
#include "cub/block/block_scan.cuh"

__global__
void serialNeigborGatheringKernel(int noVertices, int* cAdjacencyList, int* rAdjacencyList, int* inVertices, int* outEdges, int* outCounter, int iteration)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if (thid < noVertices)
	{
		int vertex = inVertices[thid];
		int queueIndex = atomicAdd(outCounter, rAdjacencyList[vertex + 1] - rAdjacencyList[vertex]);
		for (int offset = rAdjacencyList[vertex]; offset < rAdjacencyList[vertex + 1]; offset++)
		{
			outEdges[queueIndex] = cAdjacencyList[offset];
			queueIndex++;
		}
	}
}

__global__
void warpBasedNeighborGatheringKernel(int noVertices, int* cAdjacencyList, int* rAdjacencyList, int* inVertices, int* outEdges, int* outCounter)
{
	__shared__ volatile int command[256 / 32][4];
	int warpId = threadIdx.x / 32;
	if (warpId * 32 < noVertices)
	{
		int offsetStart, offsetEnd;
		if (blockIdx.x * blockDim.x + threadIdx.x < noVertices)
		{
			int vertex = inVertices[blockIdx.x * blockDim.x + threadIdx.x];
			offsetStart = rAdjacencyList[vertex];
			offsetEnd = rAdjacencyList[vertex + 1];
		}
		else
		{
			offsetStart = 0;
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
				command[warpId][1] = offsetStart;
				command[warpId][2] = offsetEnd;
				command[warpId][3] = atomicAdd(outCounter, offsetEnd - offsetStart);
				offsetEnd = 0;
			}
			int index = command[warpId][1] + threadIdx.x % 32;
			int adjEnd = command[warpId][2];
			while (index < adjEnd)
			{
				outEdges[command[warpId][3] + index - command[warpId][1]] = cAdjacencyList[index];
				index += 32;
			}
		}
	}
}

__global__
void CTANeighborGatheringKernel(int noVertices, int* cAdjacencyList, int* rAdjacencyList, int* inVertices, int* outEdges, int* globalSeized)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int command[256];
	int r = 0;
	int rEnd = 0;
	int neighborsBefore = 0;
	if (thid < noVertices)
	{
		int vertex = inVertices[thid];
		r = rAdjacencyList[vertex];
		rEnd = rAdjacencyList[vertex+1];
	}
	typedef cub::BlockScan<int, 256, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	__syncthreads();
	int total;
	BlockScan(temp_storage).ExclusiveSum(rEnd - r, neighborsBefore, total);
	__syncthreads();
	__shared__ volatile int blockOffset;
	if ((thid == noVertices - 1)||( threadIdx.x == 255 && thid < noVertices))
	{
		blockOffset = atomicAdd(globalSeized, total);
		printf("Total to save: %d, blockOffset: %d\n", total, blockOffset);
	}
	__syncthreads();
	int ctaProgress = 0;
	while (total > ctaProgress)
	{
		while (neighborsBefore < ctaProgress + 256 && r < rEnd)
		{
			command[neighborsBefore - ctaProgress] = r;
			neighborsBefore++;
			r++;
		}
		__syncthreads();
		if (threadIdx.x < total - ctaProgress && threadIdx.x < 256)
		{
			outEdges[blockOffset + ctaProgress + threadIdx.x] = cAdjacencyList[command[threadIdx.x]];
		}
		ctaProgress += 256;
		__syncthreads();
	}
}

__global__
void serialNeighborGatheringPrefixSumKernel(int noVertices, int* rAdjacencyList, int* inVertices, int* globalSeized, int* cAdjacencyList, int* outEdges)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if (thid < noVertices)
	{
		int neighborsBefore = 0;
		int vertex = inVertices[thid];
		int r = rAdjacencyList[vertex];
		int rEnd = rAdjacencyList[vertex + 1];

		typedef cub::BlockScan<int, 256, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;
		__syncthreads();
		BlockScan(temp_storage).ExclusiveSum(rEnd - r, neighborsBefore);
		__syncthreads();
		__shared__ volatile int blockOffset;
		if (thid == noVertices - 1 || threadIdx.x == 255)
		{
			blockOffset = atomicAdd(globalSeized, neighborsBefore + rEnd - r);
		}
		__syncthreads();
		int queueIndex = blockOffset + neighborsBefore;
		for (int i = r; i < rEnd; i++)
		{
			outEdges[queueIndex] = cAdjacencyList[i];
			queueIndex++;
		}
	}
}

__global__
void warpBasedNeighborGatheringPrefixSumKernel(int noVertices, int* rAdjacencyList, int* inVertices, int* globalSeized, int* cAdjacencyList, int* outEdges)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int r = 0, rEnd = 0, neighborsBefore = 0;
	__shared__ volatile int blockOffset;
	if (thid < noVertices)
	{
		int vertex = inVertices[thid];
		r = rAdjacencyList[vertex];
		rEnd = rAdjacencyList[vertex + 1];

		typedef cub::BlockScan<int, 256, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;
		__syncthreads();
		BlockScan(temp_storage).ExclusiveSum(rEnd - r, neighborsBefore);
		__syncthreads();
		if (thid == noVertices - 1 || threadIdx.x == 255)
		{
			blockOffset = atomicAdd(globalSeized, neighborsBefore + rEnd - r);
		}
	}
	__shared__ volatile int command[256 / 32][4];
	__syncthreads();
	int warpId = threadIdx.x / 32;
	if (warpId * 32 < noVertices)
	{
		while (1)
		{
			command[warpId][0] = -1;
			if (rEnd != 0)
			{
				command[warpId][0] = threadIdx.x;
			}
			if (command[warpId][0] == -1)
			{
				break;
			}
			if (command[warpId][0] == threadIdx.x)
			{
				command[warpId][1] = r;
				command[warpId][2] = rEnd;
				command[warpId][3] = blockOffset + neighborsBefore;
				rEnd = 0;
			}
			int index = command[warpId][1] + threadIdx.x % 32;
			int adjEnd = command[warpId][2];
			while (index < adjEnd)
			{
				outEdges[command[warpId][3] + index - command[warpId][1]] = cAdjacencyList[index];
				index += 32;
			}
		}
	}
}