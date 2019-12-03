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
#include "tester.h"
#include "ResultsToCSV.h"

float launchSolution(SOLUTION_TYPE solution, int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int* model);
float launchNvSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int* model);
float launchCPUSolutions(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int noTimes);
void showUsage(std::vector<std::string> solutions);
std::vector<std::string> GetChosenSolutions(int argc, char** argv, std::map<std::string, SOLUTION_TYPE> solutions, bool& out_cpuSolutionTesting, bool& out_nvSolutionTesting);

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
	solutions.insert(std::make_pair("CTA", SOLUTION_TYPE::CTA));
	solutions.insert(std::make_pair("CTA_dd", SOLUTION_TYPE::CTA_DD));
	solutions.insert(std::make_pair("CTA_add", SOLUTION_TYPE::CTA_ADD));

	std::vector<std::string> chosenSolutions;
	std::vector<std::string> solutionNames(solutions.size());
	int* cAdjacencyList;
	int* rAdjacencyList;
	int noVertices;
	int noEdges;
	int noTests;
	int* traversalModel;
	bool cpuSolutionTesting = false;
	bool nvSolutionTesting = false;
	float cpuTime = 0;
	float nvTime = 0;

	for (auto it = solutions.begin(); it != solutions.end(); ++it)
	{
		solutionNames[it->second] = it->first;
	}

	if (argc < 5)
	{
		showUsage(solutionNames);
		return 0;
	}
	CheckCSV(argv[2], solutionNames.data());
	noTests = strtol(argv[3], NULL, 10);

	chosenSolutions = GetChosenSolutions(argc, argv, solutions, cpuSolutionTesting, nvSolutionTesting);

	if (assemble_csr_matrix(argv[1], &rAdjacencyList, &cAdjacencyList, &noVertices, &noEdges) == -1)
	{
		showUsage(solutionNames);
		return 0;
	}
	printf("Vertices: %d\n", noVertices);
	printf("Edges: %d\n", noEdges);
	printf("Number of tests: %d\n", noTests);
	printf("Number of solutions: %zd\n", chosenSolutions.size());


	printf("Creating traversal model\n");
	traversalModel = createModel(cAdjacencyList, rAdjacencyList, noVertices, noEdges, 0);

	printf("Running tests\n");
	float* timeSums = (float*)malloc(sizeof(float) * solutions.size());
	bool* errorEncountered = (bool*)malloc(sizeof(bool)*solutions.size());
	for (int i = 0; i < solutions.size(); i++)
	{
		timeSums[i] = 0;
		errorEncountered[i] = false;
	}
	for (int i = 0; i < noTests; i++)
	{
		progressBar(i, noTests);
		for (std::string var: chosenSolutions)
		{
			if (errorEncountered[(int)solutions[var]])
				continue;
			float time = launchSolution(solutions[var], cAdjacencyList, rAdjacencyList, noVertices, noEdges, 0, traversalModel);
			if (time < 0)
			{
				errorEncountered[(int)solutions[var]] = true;
				continue;
			}
			timeSums[(int)solutions[var]] += time;
		}
		if (nvSolutionTesting)
		{
			nvTime += launchNvSolution(cAdjacencyList, rAdjacencyList, noVertices, noEdges, 0, traversalModel);
		}
	}
	progressBar(100, 100);
	printf("\n");

	if (cpuSolutionTesting)
	{
		printf("Testing CPU\n");
		cpuTime += launchCPUSolutions(cAdjacencyList, rAdjacencyList, noVertices, noEdges, 0, noTests);
	}

	for (std::string var : chosenSolutions)
	{
		timeSums[(int)solutions[var]] /= noTests;
	}
	nvTime /= noTests;
	cpuTime /= noTests;

	printf("Saving to file\n");
	SaveResults(argv[2], argv[1], noVertices, noEdges, noTests, nvTime, cpuTime, solutionNames.data(), timeSums, errorEncountered, (int)solutions.size());

	printf("\nResults\n");
	if (nvSolutionTesting)
	{
		printf("%25s = %11f\n", "nvgraph", nvTime);
	}
	for (std::string var : chosenSolutions)
	{
		if(errorEncountered[(int)solutions[var]])
			printf("%25s = ERROR\n", var.c_str());
		else
			printf("%25s = %11f\n", var.c_str(), timeSums[(int)solutions[var]]);
	}
	if (cpuSolutionTesting)
	{
		printf("%25s = %11f\n", "CPU", cpuTime);
	}

	free(cAdjacencyList);
	free(rAdjacencyList);
	return 0;
}


float launchSolution(SOLUTION_TYPE solution, int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int* model)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int* d_dist;
	d_dist = solutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, solution);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	int* dist;
	dist = (int*)malloc(noVertices * sizeof(int));
	gpuErrchk(cudaMemcpy(dist, d_dist, noVertices * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(d_dist));
	if (!testSolution(noVertices, dist, model))
	{
		return (float)-1;
	}
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

float launchNvSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int* model)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	nvgraphHandle_t handle;
	int* bfs_distances_h = (int*)malloc(sizeof(int)*noVertices);
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
	vertex_dimT = (cudaDataType_t*)malloc(sizeof(cudaDataType_t));
	vertex_dimT[distances_index] = CUDA_R_32I;
	nvgraphAllocateVertexData(handle, descrG, 1, vertex_dimT);
	nvgraphTraversalParameterInit(&params);
	nvgraphTraversalSetDistancesIndex(&params, distances_index);
	nvgraphTraversalSetUndirectedFlag(&params, false);
	source_vect = startingVertex;

	nvgraphTraversal(handle, descrG, NVGRAPH_TRAVERSAL_BFS, &source_vect, params);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	nvgraphGetVertexData(handle, descrG, (void*)bfs_distances_h, distances_index);
	free(vertex_dimT);
	free(topologyData);
	nvgraphDestroyGraphDescr(handle, descrG);
	nvgraphDestroy(handle);
	if (!testSolution(noVertices, bfs_distances_h, model))
	{
		printf("NvGraph solution returned invalid outcome\n");
		exit(0);
	}
	return milliseconds;
}

void showUsage(std::vector<std::string> solutionNames)
{
	printf("usage:\t mybfs\t file_mtx\t number_of_tests\t solution1\t [solution2 ...]\n");
	printf("  file_mtx is a path to the graph file in the Matrix Market format\n");
	printf("  available solutions:\n");
	for (auto & var : solutionNames)
	{
		printf("    %s\n", var.c_str());
	}
}

std::vector<std::string> GetChosenSolutions(int argc, char** argv, std::map<std::string, SOLUTION_TYPE> solutions, bool& out_cpuSolutionTesting, bool& out_nvSolutionTesting)
{
	std::vector<std::string> chosenSolutions;
	if (strcmp(argv[4], "all") == 0)
	{
		for (auto var : solutions)
		{
			chosenSolutions.push_back(var.first);
		}
		out_cpuSolutionTesting = true;
		out_nvSolutionTesting = true;
	}
	else if (strcmp(argv[4], "without_scan") == 0)
	{
		for (auto var : solutions)
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
		out_cpuSolutionTesting = true;
		out_nvSolutionTesting = true;
	}
	else if (strcmp(argv[4], "without_CTA") == 0)
	{
		for (auto var : solutions)
		{
			if (var.first != "CTA" &&
				var.first != "CTA_dd" &&
				var.first != "CTA_add")
			{
				chosenSolutions.push_back(var.first);
			}
		}
		out_cpuSolutionTesting = true;
		out_nvSolutionTesting = true;
	}
	else if (strcmp(argv[4], "test_here") == 0)
	{
		for (auto var : solutions)
		{
			if (var.first != "warp_scan" &&
				var.first != "warp_scan_dd" &&
				var.first != "warp_scan_add" &&
				var.first != "warp_half_scan" &&
				var.first != "warp_half_scan_dd" &&
				var.first != "warp_half_scan_add" 
				)
			{
				chosenSolutions.push_back(var.first);
			}
		}
		out_cpuSolutionTesting = true;
		out_nvSolutionTesting = true;
	}
	else
	{
		for (int i = 4; i < argc; i++)
		{
			if (strcmp(argv[i], "CPU") == 0)
			{
				out_cpuSolutionTesting = true;
			}
			else if (strcmp(argv[i], "nvgraph") == 0)
			{
				out_nvSolutionTesting = true;
			}
			else
			{
				auto sol = solutions.find(argv[i]);
				if (sol == solutions.end())
				{
					printf("Solution %s not found \n", argv[i]);
					exit(0);
				}
				else
				{
					chosenSolutions.push_back(sol->first);
				}
			}
		}
	}
	return chosenSolutions;
}

