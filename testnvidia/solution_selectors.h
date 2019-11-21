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

int* NSquaredSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int selection)
{
	int* d_vertexFrontier, *d_vertexDistance;
	int *d_cAdjacencyList, *d_rAdjacencyList;

	gpuErrchk(cudaMalloc(&d_vertexFrontier, noVertices * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_vertexDistance, noVertices * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_cAdjacencyList, noEdges * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_rAdjacencyList, (noVertices + 1) * sizeof(int)));

	gpuErrchk(cudaMemset(d_vertexFrontier, 0, noVertices * sizeof(int)));
	gpuErrchk(cudaMemset(d_vertexFrontier + startingVertex, 1, 1));
	gpuErrchk(cudaMemset(d_vertexDistance, 1, noVertices * sizeof(int)));	//sets all ints to INF (16843009)
	gpuErrchk(cudaMemset(d_vertexDistance + startingVertex, 0, sizeof(int)));
	gpuErrchk(cudaMemcpy(d_cAdjacencyList, cAdjacencyList, noEdges * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_rAdjacencyList, rAdjacencyList, (noVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));

	int* isVertexFrontierEmpty;
	gpuErrchk(cudaMallocManaged(&isVertexFrontierEmpty, sizeof(int)));
	isVertexFrontierEmpty[0] = 0;
	int iteration = 0;

	while (isVertexFrontierEmpty[0] == 0)
	{
		isVertexFrontierEmpty[0] = 1;
		NSquaredKernel << <noVertices / NO_THREADS + 1, NO_THREADS >> > (noVertices, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, isVertexFrontierEmpty);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		iteration++;
	}

#ifdef TEST_MODE
	testSolution(noVertices, d_vertexDistance);
#endif

	gpuErrchk(cudaFree(isVertexFrontierEmpty));
	gpuErrchk(cudaFree(d_vertexFrontier));
	//cudaFree(d_vertexDistance);
	gpuErrchk(cudaFree(d_cAdjacencyList));
	gpuErrchk(cudaFree(d_rAdjacencyList));
	return d_vertexDistance;
}

int* expandContractSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, SOLUTION_TYPE solution)
{
	int inCounter = 1;
	int* d_outCounter;
	int iteration = 0;

	gpuErrchk(cudaMalloc(&d_outCounter, sizeof(int)));
	gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));

	int* d_vertexDistance;
	gpuErrchk(cudaMalloc(&d_vertexDistance, noVertices * sizeof(int)));
	gpuErrchk(cudaMemset(d_vertexDistance, 1, noVertices * sizeof(int)));	//sets all ints to INF
	gpuErrchk(cudaMemset(d_vertexDistance + startingVertex, 0, sizeof(int)));

	int* d_inVertexQueue;
	int* d_outVertexQueue;
	int *d_cAdjacencyList;
	int *d_rAdjacencyList;
	gpuErrchk(cudaMalloc(&d_inVertexQueue, noVertices * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_outVertexQueue, noVertices * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_inVertexQueue, &startingVertex, sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&d_cAdjacencyList, noEdges * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_rAdjacencyList, (noVertices + 1) * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_cAdjacencyList, cAdjacencyList, noEdges * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_rAdjacencyList, rAdjacencyList, (noVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));

	switch (solution)
	{
	case SOLUTION_TYPE::SERIAL_ATOMIC_ONE_PHASE:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			serialGatheringAtomicArrayKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, d_inVertexQueue, d_outVertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			int* buffer = d_inVertexQueue;
			d_inVertexQueue = d_outVertexQueue;
			d_outVertexQueue = buffer;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		break;
	}
	case SOLUTION_TYPE::WARP_ATOMIC_ONE_PHASE:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			warpBasedGatheringAtomicArrayKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_cAdjacencyList, d_rAdjacencyList, d_inVertexQueue, d_outVertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			int* buffer = d_inVertexQueue;
			d_inVertexQueue = d_outVertexQueue;
			d_outVertexQueue = buffer;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		break;
	}
	}
#ifdef TEST_MODE
	testSolution(noVertices, d_vertexDistance);
#endif

	gpuErrchk(cudaFree(d_outCounter));
	gpuErrchk(cudaFree(d_inVertexQueue));
	gpuErrchk(cudaFree(d_outVertexQueue));
	gpuErrchk(cudaFree(d_cAdjacencyList));
	gpuErrchk(cudaFree(d_rAdjacencyList));
	return d_vertexDistance;
}

int* twoPhaseSolutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, SOLUTION_TYPE solution)
{
	int inCounter = 1;
	int* d_outCounter;
	int iteration = 0;

	gpuErrchk(cudaMalloc(&d_outCounter, sizeof(int)));
	gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));

	int* d_vertexDistance;
	gpuErrchk(cudaMalloc(&d_vertexDistance, noVertices * sizeof(int)));
	gpuErrchk(cudaMemset(d_vertexDistance, 1, noVertices * sizeof(int)));	//sets all ints to INF
	gpuErrchk(cudaMemset(d_vertexDistance + startingVertex, 0, sizeof(int)));

	int* d_vertexQueue;
	int* d_edgeQueue;
	int *d_cAdjacencyList;
	int *d_rAdjacencyList;
	gpuErrchk(cudaMalloc(&d_vertexQueue, noVertices * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_edgeQueue, noEdges * sizeof(int)));

	gpuErrchk(cudaMemcpy(d_vertexQueue, &startingVertex, sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&d_cAdjacencyList, noEdges * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_rAdjacencyList, (noVertices + 1) * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_cAdjacencyList, cAdjacencyList, noEdges * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_rAdjacencyList, rAdjacencyList, (noVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));

	switch (solution)
	{
	case SOLUTION_TYPE::SERIAL_ATOMIC:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			serialNeigborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter, iteration);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}
	case SOLUTION_TYPE::SERIAL_ATOMIC_DD:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			serialNeigborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter, iteration);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupDuplicateDetectionKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}
	case SOLUTION_TYPE::SERIAL_ATOMIC_ADD:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			serialNeigborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter, iteration);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupADDKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}
	case SOLUTION_TYPE::SERIAL_SCAN:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_noEdgesValidsBefore, (noEdges + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			precountForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		gpuErrchk(cudaFree(d_noEdgesValidsBefore));
		break;
	}
	case SOLUTION_TYPE::SERIAL_SCAN_DD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_noEdgesValidsBefore, (noEdges + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			precountWithDuplicateDetectionForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		gpuErrchk(cudaFree(d_noEdgesValidsBefore));
		break;
	}
	case SOLUTION_TYPE::SERIAL_SCAN_ADD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_noEdgesValidsBefore, (noEdges + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			precountWithADDForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		gpuErrchk(cudaFree(d_noEdgesValidsBefore));
		break;
	}
	case SOLUTION_TYPE::SERIAL_HALF_SCAN:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		break;
	}
	case SOLUTION_TYPE::SERIAL_HALF_SCAN_DD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupDuplicateDetectionKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		break;
	}

	case SOLUTION_TYPE::SERIAL_HALF_SCAN_ADD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			serialNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupADDKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		break;
	}
	case SOLUTION_TYPE::WARP_ATOMIC:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			warpBasedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			//cudaDeviceSynchronize();
			atomicArrayLookupKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			//cudaDeviceSynchronize();
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}
	case SOLUTION_TYPE::WARP_ATOMIC_DD:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			warpBasedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupDuplicateDetectionKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}
	case SOLUTION_TYPE::WARP_ATOMIC_ADD:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			warpBasedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupADDKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}
	case SOLUTION_TYPE::WARP_SCAN:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_noEdgesValidsBefore, (noEdges + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			precountForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		gpuErrchk(cudaFree(d_noEdgesValidsBefore));
		break;
	}
	case SOLUTION_TYPE::WARP_SCAN_DD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_noEdgesValidsBefore, (noEdges + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			precountWithDuplicateDetectionForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		gpuErrchk(cudaFree(d_noEdgesValidsBefore));
		break;
	}
	case SOLUTION_TYPE::WARP_SCAN_ADD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_noEdgesValidsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_noEdgesValidsBefore, (noEdges + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			precountWithADDForScanLookupKernel << < (edgesQueueCounter) / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_noEdgesValidsBefore, d_seized);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			scanLookupKernel << < edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, d_edgeQueue, d_vertexQueue, d_noEdgesValidsBefore);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		gpuErrchk(cudaFree(d_noEdgesValidsBefore));
		break;
	}
	case SOLUTION_TYPE::WARP_HALF_SCAN:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		break;
	}
	case SOLUTION_TYPE::WARP_HALF_SCAN_DD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupDuplicateDetectionKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		break;
	}
	case SOLUTION_TYPE::WARP_HALF_SCAN_ADD:
	{
		int* d_noVertexNeighborsBefore;
		int* d_seized;

		gpuErrchk(cudaMalloc(&d_noVertexNeighborsBefore, (noVertices + 1) * sizeof(int)));
		gpuErrchk(cudaMalloc(&d_seized, sizeof(int)));
		gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));

		int edgesQueueCounter = 0;

		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_seized, 0, sizeof(int)));
			warpBasedNeighborGatheringPrefixSumKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_rAdjacencyList, d_vertexQueue, d_seized, d_cAdjacencyList, d_edgeQueue);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&edgesQueueCounter, d_seized, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupADDKernel << <edgesQueueCounter / NO_THREADS + 1, NO_THREADS >> > (edgesQueueCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		gpuErrchk(cudaFree(d_seized));
		gpuErrchk(cudaFree(d_noVertexNeighborsBefore));
		break;
	}
	case SOLUTION_TYPE::CTA:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			CTANeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		break;
	}
	case SOLUTION_TYPE::CTA_DD:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			CTANeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupDuplicateDetectionKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		break;
	}
	case SOLUTION_TYPE::CTA_ADD:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			CTANeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupADDKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			iteration++;
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
		}
		break;
	}
	case SOLUTION_TYPE::WARP_BOOSTED:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			warpBoostedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}
	case SOLUTION_TYPE::WARP_BOOSTED_DD:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			warpBoostedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupDuplicateDetectionKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}
	case SOLUTION_TYPE::WARP_BOOSTED_ADD:
	{
		while (inCounter != 0)
		{
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			warpBoostedNeighborGatheringKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, d_cAdjacencyList, d_rAdjacencyList, d_vertexQueue, d_edgeQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemset(d_outCounter, 0, sizeof(int)));
			atomicArrayLookupADDKernel << <inCounter / NO_THREADS + 1, NO_THREADS >> > (inCounter, iteration, d_vertexDistance, d_edgeQueue, d_vertexQueue, d_outCounter);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaMemcpy(&inCounter, d_outCounter, sizeof(int), cudaMemcpyDeviceToHost));
			iteration++;
		}
		break;
	}

	}
#ifdef TEST_MODE
	testSolution(noVertices, d_vertexDistance);
#endif

	gpuErrchk(cudaFree(d_outCounter));
	gpuErrchk(cudaFree(d_vertexQueue));
	gpuErrchk(cudaFree(d_edgeQueue));
	gpuErrchk(cudaFree(d_cAdjacencyList));
	gpuErrchk(cudaFree(d_rAdjacencyList));
	return d_vertexDistance;
}

int* solutionSelector(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, SOLUTION_TYPE solution)
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