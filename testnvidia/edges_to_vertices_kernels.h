#pragma once
#include "defines.h"
#include "cub/block/block_scan.cuh"

__global__
void atomicArrayLookupKernel(int noEdges, int iteration, int* dist, int* inEdges, int* outVertices, int* outCounter)
{

	//if(threadIdx.x == 0 && iteration < 4)
	//	printf("Na wejsciu lookupu mam %d krawedzi\n", noEdges);
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if (thid < noEdges)
	{
		//if (iteration < 4)
			//printf("Odleglosc do wierzcholka %d: %d\n", inEdges[blockIdx.x * blockDim.x + threadIdx.x], dist[inEdges[blockIdx.x * blockDim.x + threadIdx.x]]);
		if (dist[inEdges[thid]] == INF)
		{
			dist[inEdges[thid]] = iteration + 1;
			int queueIndex = atomicAdd(outCounter, 1);
			//if(iteration == 5)
			//	printf("Zapisuje krawedz %d jako wierzcholek\n", inEdges[blockIdx.x * blockDim.x + threadIdx.x]);
			outVertices[queueIndex] = inEdges[thid];
		}
	}
}

__global__
void atomicArrayLookupDuplicateDetectionKernel(int noEdges, int iteration, int* dist, int* inEdges, int* outVertices, int* outCounter)
{
	__shared__ volatile int scratch[32][128];
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpid = threadIdx.x >> 5;
	if (thid < noEdges)
	{
		if (dist[inEdges[thid]] == INF)
		{
			int hash = inEdges[thid] & 127;
			scratch[warpid][hash] = inEdges[thid];
			if (scratch[warpid][hash] == inEdges[thid])
			{
				scratch[warpid][hash] = thid;
				if (scratch[warpid][hash] == thid)
				{
					dist[inEdges[thid]] = iteration + 1;
					int queueIndex = atomicAdd(outCounter, 1);
					outVertices[queueIndex] = inEdges[thid];
				}
			}
			else
			{
				dist[inEdges[thid]] = iteration + 1;
				int queueIndex = atomicAdd(outCounter, 1);
				outVertices[queueIndex] = inEdges[thid];
			}
		}
	}
}

__global__
void atomicArrayLookupADDKernel(int noEdges, int iteration, int* dist, int* inEdges, int* outVertices, int* outCounter)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if (thid < noEdges)
	{
		if (dist[inEdges[thid]] == INF)
		{
			int stillInf = atomicExch(&(dist[inEdges[thid]]), iteration + 1);
			if (stillInf == INF)
			{
				int queueIndex = atomicAdd(outCounter, 1);
				outVertices[queueIndex] = inEdges[thid];
			}
		}
	}
}

__global__
void precountForScanLookupKernel(int noEdges, int iteration, int* dist, int* inEdges, int* globalEdgesValidsBefore, int* globalSeized)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if (thid < noEdges)
	{
		int edgeValid= 0;
		int edgesValidBefore = 0;
		if (dist[inEdges[thid]] == INF)
		{

			dist[inEdges[thid]] = iteration + 1;
			edgeValid = 1;
		}
		typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
		// Allocate shared memory for BlockScan
		__shared__ typename BlockScan::TempStorage temp_storage;
		// Obtain a segment of consecutive items that are blocked across threads
		// Collectively compute the block-wide exclusive prefix sum
		__syncthreads();
		BlockScan(temp_storage).ExclusiveSum(edgeValid, edgesValidBefore);
		__syncthreads();
		__shared__ volatile int blockOffset;
		if (thid == noEdges - 1 || threadIdx.x == 255)
		{
			blockOffset = atomicAdd(globalSeized, edgesValidBefore + edgeValid);
			//printf("Zwiekszylem globalSeized od %d do %d\n", blockOffset, blockOffset + edgesValidBefore + edgeValid);
		}
		__syncthreads();
		if (edgeValid)
		{
			globalEdgesValidsBefore[thid] = edgesValidBefore + blockOffset;
		}
		else
		{
			globalEdgesValidsBefore[thid] = -1;
		}
	}
}

__global__
void precountWithDuplicateDetectionForScanLookupKernel(int noEdges, int iteration, int* dist, int* inEdges, int* globalEdgesValidsBefore, int* globalSeized)
{
	__shared__ volatile int scratch[32][128];
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpid = threadIdx.x >> 5;
	if (thid < noEdges)
	{
		int edgeValid = 0;
		int edgesValidBefore = 0;
		if (dist[inEdges[thid]] == INF)
		{
			int hash = inEdges[thid] & 127;
			scratch[warpid][hash] = inEdges[thid];
			if (scratch[warpid][hash] == inEdges[thid])
			{
				scratch[warpid][hash] = thid;
				if (scratch[warpid][hash] == thid)
				{
					dist[inEdges[thid]] = iteration + 1;
					edgeValid = 1;
				}
			}
			else
			{
				dist[inEdges[thid]] = iteration + 1;
				edgeValid = 1;
			}
		}
		typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;
		__syncthreads();
		BlockScan(temp_storage).ExclusiveSum(edgeValid, edgesValidBefore);
		__syncthreads();
		__shared__ volatile int blockOffset;
		if (thid == noEdges - 1 || threadIdx.x == 255)
		{
			blockOffset = atomicAdd(globalSeized, edgesValidBefore + edgeValid);
		}
		__syncthreads();
		if (edgeValid)
		{
			globalEdgesValidsBefore[thid] = edgesValidBefore + blockOffset;
		}
		else
		{
			globalEdgesValidsBefore[thid] = -1;
		}
	}
}

__global__
void precountWithADDForScanLookupKernel(int noEdges, int iteration, int* dist, int* inEdges, int* globalEdgesValidsBefore, int* globalSeized)
{
	__shared__ volatile int scratch[32][128];
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if (thid < noEdges)
	{
		int edgeValid = 0;
		int edgesValidBefore = 0;
		if (dist[inEdges[thid]] == INF)
		{
			int stillInf = atomicExch(&(dist[inEdges[thid]]), iteration + 1);
			if (stillInf == INF)
			{
				edgeValid = 1;
			}
		}
		typedef cub::BlockScan<int, NO_THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScan;
		__shared__ typename BlockScan::TempStorage temp_storage;
		__syncthreads();
		BlockScan(temp_storage).ExclusiveSum(edgeValid, edgesValidBefore);
		__syncthreads();
		__shared__ volatile int blockOffset;
		if (thid == noEdges - 1 || threadIdx.x == 255)
		{
			blockOffset = atomicAdd(globalSeized, edgesValidBefore + edgeValid);
		}
		__syncthreads();
		if (edgeValid)
		{
			globalEdgesValidsBefore[thid] = edgesValidBefore + blockOffset;
		}
		else
		{
			globalEdgesValidsBefore[thid] = -1;
		}
	}
}

__global__
void scanLookupKernel(int noEdges, int* inEdges, int* outVertices, int* noEdgesValidsBefore)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if (thid < noEdges)
	{
		if (noEdgesValidsBefore[thid] != -1)
		{
			//if(iteration == 5)
			//	printf("Zapisuje krawedz %d jako wierzcholek\n", inEdges[blockIdx.x * blockDim.x + threadIdx.x]);
			//printf("Zapisuje do kolejki wierzcholkow %d na miejsce %d\n", inEdges[thid], noEdgesValidsBefore[thid]);
			outVertices[noEdgesValidsBefore[thid]] = inEdges[thid];
		}
	}
}