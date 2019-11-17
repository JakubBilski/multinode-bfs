#include <vector>
#include <map>
#include <string>
#include <utility>
#include <time.h>

#include "solution_selectors.h"
#include "cpu_solution_selector.h"
#include "MMToCSR.h"
#include "scan.cuh"
#include "nvgraph.h"

typedef void(*solutionFunction)(int*, int*, int, int, int);

float launchSolution(SOLUTION_TYPE solution, int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex);
float launchNvSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex);
float launchCPUSolutions(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int noTimes);
void showUsage(std::map<std::string, SOLUTION_TYPE> solutions);

int main(int argc, char* argv[])
{
	std::map<std::string, SOLUTION_TYPE> solutions;


	solutions.insert(std::make_pair("n_squared", SOLUTION_TYPE::N_SQUARED));

	solutions.insert(std::make_pair("serial_atomic_one_phase", SOLUTION_TYPE::SERIAL_ATOMIC_ONE_PHASE));
	solutions.insert(std::make_pair("serial_atomic", SOLUTION_TYPE::SERIAL_ATOMIC));
	solutions.insert(std::make_pair("serial_atomic_add", SOLUTION_TYPE::SERIAL_ATOMIC_ADD));
	solutions.insert(std::make_pair("serial_atomic_dd", SOLUTION_TYPE::SERIAL_ATOMIC_DD));
	solutions.insert(std::make_pair("serial_scan", SOLUTION_TYPE::SERIAL_SCAN));
	solutions.insert(std::make_pair("serial_scan_dd", SOLUTION_TYPE::SERIAL_SCAN_DD));
	solutions.insert(std::make_pair("serial_scan_add", SOLUTION_TYPE::SERIAL_SCAN_ADD));
	solutions.insert(std::make_pair("serial_half_scan", SOLUTION_TYPE::SERIAL_HALF_SCAN));
	solutions.insert(std::make_pair("serial_half_scan_dd", SOLUTION_TYPE::SERIAL_HALF_SCAN_DD));
	solutions.insert(std::make_pair("serial_half_scan_add", SOLUTION_TYPE::SERIAL_HALF_SCAN_ADD));

	solutions.insert(std::make_pair("warp_atomic_one_phase", SOLUTION_TYPE::WARP_ATOMIC_ONE_PHASE));
	solutions.insert(std::make_pair("warp_atomic", SOLUTION_TYPE::WARP_ATOMIC));
	solutions.insert(std::make_pair("warp_atomic_add", SOLUTION_TYPE::WARP_ATOMIC_ADD));
	solutions.insert(std::make_pair("warp_atomic_dd", SOLUTION_TYPE::WARP_ATOMIC_DD));
	solutions.insert(std::make_pair("warp_scan", SOLUTION_TYPE::WARP_SCAN));
	solutions.insert(std::make_pair("warp_scan_dd", SOLUTION_TYPE::WARP_SCAN_DD));
	solutions.insert(std::make_pair("warp_scan_add", SOLUTION_TYPE::WARP_SCAN_ADD));
	solutions.insert(std::make_pair("warp_half_scan", SOLUTION_TYPE::WARP_HALF_SCAN));
	solutions.insert(std::make_pair("warp_half_scan_dd", SOLUTION_TYPE::WARP_HALF_SCAN_DD));
	solutions.insert(std::make_pair("warp_half_scan_add", SOLUTION_TYPE::WARP_HALF_SCAN_ADD));

	solutions.insert(std::make_pair("warp_boosted", SOLUTION_TYPE::WARP_BOOSTED));
	solutions.insert(std::make_pair("warp_boosted_dd", SOLUTION_TYPE::WARP_BOOSTED_DD));
	solutions.insert(std::make_pair("warp_boosted_add", SOLUTION_TYPE::WARP_BOOSTED_ADD));

	//solutions.insert(std::make_pair("CTA", SOLUTION_TYPE::CTA));
	//solutions.insert(std::make_pair("CTA_dd", SOLUTION_TYPE::CTA_DD));
	//solutions.insert(std::make_pair("CTA_add", SOLUTION_TYPE::CTA_ADD));

	std::vector<std::string> chosenSolutions;

	if (argc < 4)
	{
		showUsage(solutions);
		return 0;
	}
	int* cAdjacencyList;
	int* rAdjacencyList;
	int noVertices;
	int noEdges;
	int noTests = strtol(argv[2], NULL, 10);
	bool cpuSolutionTesting = false;
	bool nvSolutionTesting = false;
	float cpuTime = 0;
	float nvTime = 0;
	if (strcmp(argv[3], "all") == 0)
	{
		for each (auto var in solutions)
		{
			chosenSolutions.push_back(var.first);
		}
		cpuSolutionTesting = true;
		nvSolutionTesting = true;
	}
	else if (strcmp(argv[3], "without_scan") == 0)
	{
		for each (auto var in solutions)
		{
			if (var.first != "warp_scan" &&
				var.first != "warp_scan_dd" &&
				var.first != "warp_scan_add" &&
				var.first != "serial_scan" &&
				var.first != "serial_scan_dd" &&
				var.first != "serial_scan_add")
			{
				chosenSolutions.push_back(var.first);
			}
		}
		cpuSolutionTesting = true;
		nvSolutionTesting = true;
	}
	else
	{
		for (int i = 3; i < argc; i++)
		{
			if (strcmp(argv[i], "CPU") == 0)
			{
				cpuSolutionTesting = true;
			}
			else if (strcmp(argv[i], "nvgraph") == 0)
			{
				nvSolutionTesting = true;
			}
			else
			{
				auto sol = solutions.find(argv[i]);
				if (sol == solutions.end())
				{
					printf("Solution %s not found \n", argv[i]);
					showUsage(solutions);
					return 0;
				}
				else
				{
					chosenSolutions.push_back(sol->first);
				}
			}
		}
	}
	if (assemble_csr_matrix(argv[1], &rAdjacencyList, &cAdjacencyList, &noVertices, &noEdges) == -1)
	{
		showUsage(solutions);
		return 0;
	}
	printf("Vertices: %d\n", noVertices);
	printf("Edges: %d\n", noEdges);
	printf("Number of tests: %d\n", noTests);
	printf("Number of solutions: %d\n", chosenSolutions.size());

	float* timeSums = (float*)malloc(sizeof(float) * chosenSolutions.size());
	for (int i = 0; i < chosenSolutions.size(); i++)
	{
		timeSums[i] = 0;
	}
	printf("Running tests\n");
#ifdef TEST_MODE
	noTests = 1;
#endif
	int index = 0;
	for (int i = 0; i < noTests; i++)
	{
		progressBar(i, noTests);
		for each (std::string var in chosenSolutions)
		{
			timeSums[index] += launchSolution(solutions[var], cAdjacencyList, rAdjacencyList, noVertices, noEdges, 0);
			index++;
		}
		if (nvSolutionTesting)
		{
			nvTime += launchNvSolution(cAdjacencyList, rAdjacencyList, noVertices, noEdges, 0);
		}
		index = 0;
	}
	progressBar(100, 100);
	printf("\n");

	if (cpuSolutionTesting)
	{
		printf("Testing CPU\n");
		cpuTime += launchCPUSolutions(cAdjacencyList, rAdjacencyList, noVertices, noEdges, 0, noTests);
	}
	printf("\n");
	printf("\nResults\n");
	if (nvSolutionTesting)
	{
		printf("%25s = %11f\n", "nvgraph", nvTime / noTests);
	}
	for each (std::string var in chosenSolutions)
	{
		if(timeSums[index] >= INF)
			printf("%25s = TIMEOUT\n", var.c_str());
		else
			printf("%25s = %11f\n", var.c_str(), timeSums[index] / noTests);
		index++;
	}
	if (cpuSolutionTesting)
	{
		printf("%25s = %11f\n", "CPU", cpuTime / noTests);
	}
	free(cAdjacencyList);
	free(rAdjacencyList);
	return 0;
}


float launchSolution(SOLUTION_TYPE solution, int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int timeout=0;
	timeout = solutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, solution);
	if (timeout)
	{
		printf("Timeout\n");
		return INF;
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	return milliseconds;
}

float launchCPUSolutions(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int noTimes)
{
	 clock_t start, end;
	 start = clock();

	 for(int i=0; i<noTimes; ++i)
		 CPUSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 0);

	 end = clock();
	 return (float(1000 * (end - start))) / CLOCKS_PER_SEC;
}

float launchNvSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	nvgraphHandle_t handle;
	nvgraphGraphDescr_t descrG;
	nvgraphCSRTopology32I_t topologyData;
	int source_vect;
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
	//size_t predecessors_index = 1;
	vertex_dimT = (cudaDataType_t*)malloc(sizeof(cudaDataType_t));
	vertex_dimT[distances_index] = CUDA_R_32I;
	//vertex_dimT[predecessors_index] = CUDA_R_32I;
	nvgraphAllocateVertexData(handle, descrG, 1, vertex_dimT);
	nvgraphTraversalParameterInit(&params);
	nvgraphTraversalSetDistancesIndex(&params, distances_index);
	//nvgraphTraversalSetPredecessorsIndex(&params, predecessors_index);
	nvgraphTraversalSetUndirectedFlag(&params, false);
	source_vect = startingVertex;

	nvgraphTraversal(handle, descrG, NVGRAPH_TRAVERSAL_BFS, &source_vect, params);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	return milliseconds;
}

void showUsage(std::map<std::string, SOLUTION_TYPE> solutions)
{
	printf("usage:\t mybfs\t file_mtx\t number_of_tests\t solution1\t [solution2 ...]\n");
	printf("  file_mtx is a path to the graph file in the Matrix Market format\n");
	printf("  available solutions:\n");
	for each (std::pair< std::string, SOLUTION_TYPE> var in solutions)
	{
		printf("    %s\n", var.first.c_str());
	}
}



