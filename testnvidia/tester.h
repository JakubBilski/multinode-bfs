#pragma once
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "defines.h"

int testSolution(int noVertices, int* vertexDistance, int* model)
{
	int matchWithModel = 1;
	for (int i = 0; i < noVertices; i++)
	{
		if (model[i] != vertexDistance[i])
		{
			matchWithModel = 0;
			break;
		}
	}
	if(!matchWithModel)
		printf("Do not match with the model!\n");
	free(vertexDistance);
	return matchWithModel;
}

int* createModel(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t descrG;
	nvgraphCSRTopology32I_t topologyData;
	int source_vect;
	int* model = (int*)malloc(sizeof(int)*noVertices);
	nvgraphTraversalParameter_t params;
	nvgraphCreate(&handle);
	nvgraphCreateGraphDescr(handle, &descrG);
	topologyData = (nvgraphCSRTopology32I_t)malloc(sizeof(nvgraphCSRTopology32I_st));
	topologyData->nvertices = noVertices;
	topologyData->nedges = noEdges;
	topologyData->source_offsets = rAdjacencyList;
	topologyData->destination_indices = cAdjacencyList;
	nvgraphSetGraphStructure(handle, descrG, topologyData, NVGRAPH_CSR_32);
	cudaDataType_t* vertex_dimT;
	size_t distances_index = 0;
	vertex_dimT = (cudaDataType_t*)malloc(sizeof(cudaDataType_t));
	vertex_dimT[distances_index] = CUDA_R_32I;
	nvgraphAllocateVertexData(handle, descrG, 1, vertex_dimT);
	nvgraphTraversalParameterInit(&params);
	nvgraphTraversalSetDistancesIndex(&params, distances_index);
	nvgraphTraversalSetUndirectedFlag(&params, false);
	source_vect = startingVertex;

	nvgraphTraversal(handle, descrG, NVGRAPH_TRAVERSAL_BFS, &source_vect, params);
	nvgraphGetVertexData(handle, descrG, (void*)model, distances_index);
	return model;
}