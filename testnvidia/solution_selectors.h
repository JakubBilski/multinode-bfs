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

void testSolution(int noVertices, int* d_vertexDistance)
{
	int kkt_powerModel[35] = { 1, 9, 73, 949, 1011, 4743, 9672, 15270, 34598, 82370, 120866, 135100, 141362, 153338,
								168056, 162231, 136142, 108148, 99471, 105297, 113890, 116618, 97062, 74982, 56588,
								45198, 35334, 25236, 15019, 4560, 300, 0, 0, 0, 0};
	//int cage10Model[25] = { 1, 4, 8, 22, 48, 85, 169, 282, 464, 728, 1082, 1599, 2130, 2301, 1274, 636, 344, 124,
	//						64, 16, 0, 0, 0, 0 };

	int* model = kkt_powerModel;
	int debug_maxIteration = 35;
	int* vertexDistance = (int*)malloc(noVertices * sizeof(int));
	cudaMemcpy(vertexDistance, d_vertexDistance, noVertices * sizeof(int), cudaMemcpyDeviceToHost);
	int* counter = (int*)malloc(sizeof(int) * debug_maxIteration);
	for (int i = 0; i < debug_maxIteration; i++)
	{
		counter[i] = 0;
	}
	for (int i = 0; i < noVertices; i++)
	{
		if (vertexDistance[i] == INF)
		{
			printf("Nie odwiedzono %d\n", i);
			counter[0]++;
		}
		else
		{
			counter[vertexDistance[i]]++;
		}
		//printf("%d: %d\t", i, vertexDistance[i]);
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
	printf("Number of not visited vertices: %d \n", counter[0]-1);
	free(vertexDistance);
	free(counter);
}



void NSquaredSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int selection)
{
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
	isVertexFrontierEmpty[0] = 0;
	int iteration = 0;

	while (isVertexFrontierEmpty[0] == 0)
	{
		isVertexFrontierEmpty[0] = 1;
		NSquaredKernel << <noVertices / 256 + 1, 256 >> > (noVertices, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, isVertexFrontierEmpty);
		cudaDeviceSynchronize();
		iteration++;
	}

#ifdef TEST_MODE
	testSolution(noVertices, d_vertexDistance);
#endif

	cudaFree(isVertexFrontierEmpty);
	cudaFree(d_vertexFrontier);
	cudaFree(d_vertexDistance);
	cudaFree(d_cAdjacencyList);
	cudaFree(d_rAdjacencyList);
}

void expandContractSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int selection)
{
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

	if (selection == 1)
	{
		while (inCounter != 0)
		{
			//printf("Iteration %d, vertices: %d\n", iteration, inCounter);
			cudaMemset(d_outCounter, 0, sizeof(int));
			serialGatheringAtomicArrayKernel << <inCounter / 256 + 1, 256 >> > (inCounter, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, d_inVertexQueue, d_outVertexQueue, d_outCounter);
			cudaDeviceSynchronize();
			iteration++;
			int* buffer = d_inVertexQueue;
			d_inVertexQueue = d_outVertexQueue;
			d_outVertexQueue = buffer;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	else if (selection == 2)
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBasedGatheringAtomicArrayKernel << <inCounter / 256 + 1, 256 >> > (inCounter, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, d_inVertexQueue, d_outVertexQueue, d_outCounter);
			cudaDeviceSynchronize();
			iteration++;
			int* buffer = d_inVertexQueue;
			d_inVertexQueue = d_outVertexQueue;
			d_outVertexQueue = buffer;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
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
}

void twoPhaseSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int selection)
{
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


	if (selection == 1)
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			serialNeigborGatheringKernel << <inCounter / 256 + 1, 256 >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter, iteration);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(d_outCounter, 0, sizeof(int));
			cudaDeviceSynchronize();
			atomicArrayLookupKernel << <inCounter / 256 + 1, 256 >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			cudaDeviceSynchronize();
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			iteration++;
		}
	}
	else if (selection == 2)
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBasedNeighborGatheringKernel << <inCounter / 256 + 1, 256 >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupKernel << <inCounter / 256 + 1, 256 >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	else if (selection == 9)
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			warpBasedNeighborGatheringKernel << <inCounter / 256 + 1, 256 >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupADDKernel << <inCounter / 256 + 1, 256 >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	else if (selection == 3 || selection == 4 || selection == 5 || selection == 6)
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
			precountForNeighborGatheringPrefixSumKernel << < (inCounter) / 256 + 1, 256 >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_noVertexNeighborsBefore, d_seized, d_cAdjacencyList);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			if(selection == 3 || selection == 5)
				serialNeighborGatheringPrefixSumKernel << <inCounter / 256 + 1, 256 >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_noVertexNeighborsBefore);
			else if(selection==4 || selection == 6)
				warpBasedNeighborGatheringPrefixSumKernel << <inCounter / 256 + 1, 256 >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_noVertexNeighborsBefore);
			cudaMemset(d_seized, 0, sizeof(int));
			cudaDeviceSynchronize();
			if(selection == 3 || selection == 4)
				precountForScanLookupKernel << < (edgesQueueCounter) / 256 + 1, 256 >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			else if(selection == 5 || selection == 6)
				precountWithDuplicateDetectionForScanLookupKernel << < (edgesQueueCounter) / 256 + 1, 256 >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			scanLookupKernel << < edgesQueueCounter / 256 + 1, 256 >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			cudaDeviceSynchronize();
			iteration++;
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
		cudaFree(d_noEdgesValidsBefore);
	}
	else if (selection == 7 || selection == 8 || selection == 10 || selection == 11)
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
			precountForNeighborGatheringPrefixSumKernel << < (inCounter) / 256 + 1, 256 >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_noVertexNeighborsBefore, d_seized, d_cAdjacencyList);
			cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			if (selection == 7 || selection == 10)
				serialNeighborGatheringPrefixSumKernel << <inCounter / 256 + 1, 256 >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_noVertexNeighborsBefore);
			else if (selection == 8 || selection == 11)
				warpBasedNeighborGatheringPrefixSumKernel << <inCounter / 256 + 1, 256 >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_noVertexNeighborsBefore);
			cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			if(selection == 7 || selection == 8)
				atomicArrayLookupKernel << <edgesQueueCounter / 256 + 1, 256 >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			else if(selection == 10 || selection == 11)
				atomicArrayLookupADDKernel << <edgesQueueCounter / 256 + 1, 256 >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
		}
		cudaFree(d_seized);
		cudaFree(d_noVertexNeighborsBefore);
	}
	else if (selection == 12)
	{
		while (inCounter != 0)
		{
			cudaMemset(d_outCounter, 0, sizeof(int));
			CTANeighborGatheringKernel << <inCounter / 256 + 1, 256 >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaMemset(d_outCounter, 0, sizeof(int));
			atomicArrayLookupKernel << <inCounter / 256 + 1, 256 >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			cudaDeviceSynchronize();
			iteration++;
			cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost);
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
}