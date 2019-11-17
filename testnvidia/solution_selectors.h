#pragma once
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "vertices_to_vertices_kernels.h"
#include "vertices_to_edges_kernels.h"
#include "edges_to_vertices_kernels.h"
#include "edges_to_edges_kernels.h"

enum SOLUTION_TYPE
{
	N_SQUARED,
	SERIAL_ATOMIC_ONE_PHASE,
	SERIAL_ATOMIC,
	SERIAL_ATOMIC_DD,
	SERIAL_ATOMIC_ADD,
	SERIAL_SCAN,
	SERIAL_SCAN_DD,
	SERIAL_SCAN_ADD,
	SERIAL_HALF_SCAN,
	SERIAL_HALF_SCAN_DD,
	SERIAL_HALF_SCAN_ADD,
	WARP_ATOMIC_ONE_PHASE,
	WARP_ATOMIC,
	WARP_ATOMIC_DD,
	WARP_ATOMIC_ADD,
	WARP_SCAN,
	WARP_SCAN_DD,
	WARP_SCAN_ADD,
	WARP_HALF_SCAN,
	WARP_HALF_SCAN_DD,
	WARP_HALF_SCAN_ADD,
	CTA,
	CTA_DD,
	CTA_ADD,
	WARP_BOOSTED,
	WARP_BOOSTED_DD,
	WARP_BOOSTED_ADD
};

void testSolution(int noVertices, int* d_vertexDistance, int fromDevice=1)
{
	int kkt_powerModel[35] = { 1, 9, 73, 949, 1011, 4743, 9672, 15270, 34598, 82370, 120866, 135100, 141362, 153338,
								168056, 162231, 136142, 108148, 99471, 105297, 113890, 116618, 97062, 74982, 56588,
								45198, 35334, 25236, 15019, 4560, 300, 0, 0, 0, 0};
	int cage10Model[25] = { 1, 4, 8, 22, 48, 85, 169, 282, 464, 728, 1082, 1599, 2130, 2301, 1274, 636, 344, 124,
							64, 16, 16, 0, 0, 0, 0 };
	int coPapersCiteseerModel[30] = { 1, 136, 109, 1002, 5001, 28936, 123198, 159570, 77192, 26536, 8406, 2507,
									849, 426, 140, 55, 7, 7, 4, 4, 1, 3, 2, 3, 6, 1, 0, 0, 0, 0};

	int* model = coPapersCiteseerModel;
	int debug_maxIteration = 30;
	int* vertexDistance;
	if (fromDevice)
	{
		vertexDistance = (int*)malloc(noVertices * sizeof(int));
		cudaMemcpy(vertexDistance, d_vertexDistance, noVertices * sizeof(int), cudaMemcpyDeviceToHost);
	}
	else
	{
		vertexDistance = d_vertexDistance;
	}
	
	int* counter = (int*)malloc(sizeof(int) * debug_maxIteration);
	for (int i = 0; i < debug_maxIteration; i++)
	{
		counter[i] = 0;
	}
	for (int i = 0; i < noVertices; i++)
	{
		if (vertexDistance[i] == INF)
		{
			counter[0]++;
		}
		else
		{
			counter[vertexDistance[i]]++;
		}
	}
	int matchWithModel = 1;
	for (int i = 0; i < debug_maxIteration; i++)
	{
		printf("\nDistance %d: %d", i, counter[i]);
		if (model[i] != counter[i])
			matchWithModel = 0;
	}
	printf("\n");
	if(!matchWithModel)
		printf("Do not match with the model!\n");
	else
		printf("Matches with the model!\n");
	printf("Number of not visited vertices: %d \n", counter[0]-1);
	free(vertexDistance);
	free(counter);
}


int NSquaredSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int selection)
{
	int result = 0;
	int* d_vertexFrontier, *d_vertexDistance;
	int *d_cAdjacencyList, *d_rAdjacencyList;

	cudaMalloc(&d_vertexFrontier, noVertices * sizeof(int));
	cudaMalloc(&d_vertexDistance, noVertices * sizeof(int));
	cudaMalloc(&d_cAdjacencyList, noEdges * sizeof(int));
	cudaMalloc(&d_rAdjacencyList, (noVertices + 1) * sizeof(int));

	cudaMemset(d_vertexFrontier, 0, noVertices * sizeof(int));
	cudaMemset(d_vertexFrontier + startingVertex, 1, 1);
	cudaMemset(d_vertexDistance, 1, noVertices * sizeof(int));	//sets all ints to INF (16843009)
	cudaMemset(d_vertexDistance + startingVertex, 0, sizeof(int));
	cudaMemcpy(d_cAdjacencyList, cAdjacencyList, noEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rAdjacencyList, rAdjacencyList, (noVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);

	int* isVertexFrontierEmpty;
	cudaMallocManaged(&isVertexFrontierEmpty, sizeof(int));
	if (cudaGetLastError() != cudaSuccess) return -1;
	isVertexFrontierEmpty[0] = 0;
	int iteration = 0;

	while (isVertexFrontierEmpty[0] == 0)
	{
		isVertexFrontierEmpty[0] = 1;
		NSquaredKernel << <noVertices / NO_THREADS + 1, NO_THREADS >> > (noVertices, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, isVertexFrontierEmpty);
		cudaDeviceSynchronize();
		iteration++;
		if (cudaGetLastError() != cudaSuccess)
		{
			result = -1;
			break;
		}
	}

#ifdef TEST_MODE
	testSolution(noVertices, d_vertexDistance);
#endif

	cudaFree(isVertexFrontierEmpty);
	cudaFree(d_vertexFrontier);
	cudaFree(d_vertexDistance);
	cudaFree(d_cAdjacencyList);
	cudaFree(d_rAdjacencyList);
	return result;
}

int expandContractSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, SOLUTION_TYPE solution)
{
	int result = 0;
	int inCounter = 1;
	int* d_outCounter;
	int iteration = 0;

	cudaMalloc(&d_outCounter, sizeof(int));
	cudaMemset(d_outCounter, 0, sizeof(int));

	int* d_vertexDistance;
	cudaMalloc(&d_vertexDistance, noVertices * sizeof(int));
	cudaMemset(d_vertexDistance, 1, noVertices * sizeof(int));	//sets all ints to INF
	cudaMemset(d_vertexDistance + startingVertex, 0, sizeof(int));

	int* d_inVertexQueue;
	int* d_outVertexQueue;
	int *d_cAdjacencyList;
	int *d_rAdjacencyList;
	cudaMalloc(&d_inVertexQueue, noVertices * sizeof(int));
	cudaMalloc(&d_outVertexQueue, noVertices * sizeof(int));
	cudaMemcpy(d_inVertexQueue, &startingVertex, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_cAdjacencyList, noEdges * sizeof(int));
	cudaMalloc(&d_rAdjacencyList, (noVertices + 1) * sizeof(int));
	cudaMemcpy(d_cAdjacencyList, cAdjacencyList, noEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rAdjacencyList, rAdjacencyList, (noVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);

	switch (solution)
	{
	case SOLUTION_TYPE::SERIAL_ATOMIC_ONE_PHASE:
	{
		while (inCounter != 0)
		{
			//printf("Iteration %d, vertices: %d\n", iteration, inCounter);
			cudaMemset(d_outCounter, 0, sizeof(int));
			serialGatheringAtomicArrayKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, d_inVertexQueue, d_outVertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			iteration++;
			int* buffer = d_inVertexQueue;
			d_inVertexQueue = d_outVertexQueue;
			d_outVertexQueue = buffer;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::WARP_ATOMIC_ONE_PHASE:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBasedGatheringAtomicArrayKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, d_inVertexQueue, d_outVertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			iteration++;
			int* buffer = d_inVertexQueue;
			d_inVertexQueue = d_outVertexQueue;
			d_outVertexQueue = buffer;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
		}
		}
		break;
	}
	}
#ifdef TEST_MODE
	testSolution(noVertices, d_vertexDistance);
#endif

	cudaFree(d_outCounter);
	cudaFree(d_vertexDistance);
	cudaFree(d_inVertexQueue);
	cudaFree(d_outVertexQueue);
	cudaFree(d_cAdjacencyList);
	cudaFree(d_rAdjacencyList);
	return result;
}

int twoPhaseSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, SOLUTION_TYPE solution)
{
	int result = 0;
	int inCounter = 1;
	int* d_outCounter;
	int iteration = 0;

	cudaMalloc(&d_outCounter, sizeof(int));
	cudaMemset(d_outCounter, 0, sizeof(int));

	int* d_vertexDistance;
	cudaMalloc(&d_vertexDistance, noVertices * sizeof(int));
	cudaMemset(d_vertexDistance, 1, noVertices * sizeof(int));	//sets all ints to INF
	cudaMemset(d_vertexDistance + startingVertex, 0, sizeof(int));

	int* d_vertexQueue;
	int* d_edgeQueue;
	int *d_cAdjacencyList;
	int *d_rAdjacencyList;
	cudaMalloc(&d_vertexQueue, noVertices * sizeof(int));
	cudaMalloc(&d_edgeQueue, noEdges * sizeof(int));

	cudaMemcpy(d_vertexQueue, &startingVertex, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_cAdjacencyList, noEdges * sizeof(int));
	cudaMalloc(&d_rAdjacencyList, (noVertices + 1) * sizeof(int));
	cudaMemcpy(d_cAdjacencyList, cAdjacencyList, noEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rAdjacencyList, rAdjacencyList, (noVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);

	switch (solution)
	{
	case SOLUTION_TYPE::SERIAL_ATOMIC:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			serialNeigborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter, iteration);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::SERIAL_ATOMIC_DD:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			serialNeigborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter, iteration);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			//cudaDeviceSynchronize();
			atomicArrayLookupDuplicateDetectionKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::SERIAL_ATOMIC_ADD:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			serialNeigborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter, iteration);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			//cudaDeviceSynchronize();
			atomicArrayLookupADDKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::SERIAL_SCAN:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_noEdgesValidsBefore, (noEdges * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_seized, 0, sizeof(int));
			//cudaDeviceSynchronize();
			precountForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			//cudaDeviceSynchronize();
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		cudaFree(d_noEdgesValidsBefore);
		break;
	}
	case SOLUTION_TYPE::SERIAL_SCAN_DD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_noEdgesValidsBefore, (noEdges * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_seized, 0, sizeof(int));
			//cudaDeviceSynchronize();
			precountWithDuplicateDetectionForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			//cudaDeviceSynchronize();
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		cudaFree(d_noEdgesValidsBefore);
		break;
	}
	case SOLUTION_TYPE::SERIAL_SCAN_ADD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_noEdgesValidsBefore, (noEdges * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_seized, 0, sizeof(int));
			//cudaDeviceSynchronize();
			precountWithADDForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			//cudaDeviceSynchronize();
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		cudaFree(d_noEdgesValidsBefore);
		break;
	}
	case SOLUTION_TYPE::SERIAL_HALF_SCAN:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		break;
	}
	case SOLUTION_TYPE::SERIAL_HALF_SCAN_DD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupDuplicateDetectionKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		break;
	}

	case SOLUTION_TYPE::SERIAL_HALF_SCAN_ADD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupADDKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		break;
	}
	case SOLUTION_TYPE::WARP_ATOMIC:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBasedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			//cudaDeviceSynchronize();
			atomicArrayLookupKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::WARP_ATOMIC_DD:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBasedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			//cudaDeviceSynchronize();
			atomicArrayLookupDuplicateDetectionKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::WARP_ATOMIC_ADD:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBasedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			//cudaDeviceSynchronize();
			atomicArrayLookupADDKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::WARP_SCAN:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_noEdgesValidsBefore, (noEdges * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_seized, 0, sizeof(int));
			//cudaDeviceSynchronize();
			precountForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			//cudaDeviceSynchronize();
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		cudaFree(d_noEdgesValidsBefore);
		break;
	}
	case SOLUTION_TYPE::WARP_SCAN_DD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_noEdgesValidsBefore, (noEdges * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_seized, 0, sizeof(int));
			//cudaDeviceSynchronize();
			precountWithDuplicateDetectionForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			//cudaDeviceSynchronize();
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		cudaFree(d_noEdgesValidsBefore);
		break;
	}
	case SOLUTION_TYPE::WARP_SCAN_ADD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_noEdgesValidsBefore, (noEdges * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_seized, 0, sizeof(int));
			//cudaDeviceSynchronize();
			precountWithADDForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			//cudaDeviceSynchronize();
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		cudaFree(d_noEdgesValidsBefore);
		break;
	}
	case SOLUTION_TYPE::WARP_HALF_SCAN:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		break;
	}
	case SOLUTION_TYPE::WARP_HALF_SCAN_DD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupDuplicateDetectionKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		break;
	}
	case SOLUTION_TYPE::WARP_HALF_SCAN_ADD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		cudaMalloc(&d_noVertexNeighborsBefore, (noVertices * 3 + 1) * sizeof(int));
		cudaMalloc(&d_seized, sizeof(int));
		cudaMemset(d_seized, 0, sizeof(int));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			cudaMemset(d_seized, 0, sizeof(int));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupADDKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		break;
	}
	case SOLUTION_TYPE::CTA:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			CTANeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);

			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::CTA_DD:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			CTANeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupDuplicateDetectionKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);

			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::CTA_ADD:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			CTANeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupADDKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);

			//cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::WARP_BOOSTED:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBoostedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			//cudaDeviceSynchronize();
			atomicArrayLookupKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::WARP_BOOSTED_DD:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBoostedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			//cudaDeviceSynchronize();
			atomicArrayLookupDuplicateDetectionKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			//cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}
	case SOLUTION_TYPE::WARP_BOOSTED_ADD:
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBoostedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupADDKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
			if (cudaGetLastError() != cudaSuccess)
			{
				result = -1;
				break;
			}
		}
		break;
	}

	}
#ifdef TEST_MODE
	testSolution(noVertices, d_vertexDistance);
#endif

	cudaFree(d_outCounter);
	cudaFree(d_vertexDistance);
	cudaFree(d_vertexQueue);
	cudaFree(d_edgeQueue);
	cudaFree(d_cAdjacencyList);
	cudaFree(d_rAdjacencyList);
	return result;
}

int solutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, SOLUTION_TYPE solution)
{
	if (solution == SOLUTION_TYPE::N_SQUARED)
	{
		return NSquaredSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, solution);
	}
	else if (solution == SOLUTION_TYPE::SERIAL_ATOMIC_ONE_PHASE || solution == SOLUTION_TYPE::WARP_ATOMIC_ONE_PHASE)
	{
		return expandContractSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, solution);
	}
	else
	{
		return twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, solution);
	}
}