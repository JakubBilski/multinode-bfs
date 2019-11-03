#include <vector>
#include <map>
#include <string>
#include <utility>
#include <time.h>

#include "solution_selectors.h"
#include "cpu_solution_selector.h"
#include "MMToCSR.h"
#include "scan.cuh"

typedef void(*solutionFunction)(int*, int*, int, int, int);

float launchSolution(void(*solution)(int*, int*, int, int, int), int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex);
float launchCPUSolutions(void(*solution)(int*, int*, int, int, int), int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int noTimes);
void showUsage(std::map<std::string, solutionFunction> solutions);


void serialAtomicOnePhaseSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	expandContractSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 1);
}

void warpBasedAtomicOnePhaseSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	expandContractSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 2);
}

void warpBasedAtomicSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 2);
}

void serialAtomicSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 1);
}

void NSquaredSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	NSquaredSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 1);
}

void serialScanSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 3);
}

void warpBasedScanSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 4);
}

void serialScanDDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 5);
}

void serialScanADDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 5);
}

void warpBasedScanDDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 6);
}

void warpBasedScanADDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 6);
}

void serialHalfScanSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 7);
}

void warpBasedHalfScanSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 8);
}

void warpBasedAtomicDDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 9);
}

void warpBasedAtomicADDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 9);
}

void serialAtomicDDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 9);
}

void serialAtomicADDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 9);
}

void serialHalfScanDDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 10);
}

void serialHalfScanADDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 10);
}

void warpBasedHalfScanDDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 11);
}

void warpBasedHalfScanADDSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 11);
}

void CTATwoPhaseSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	twoPhaseSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 12);
}

void CPUSolution(int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	CPUSolutionSelector(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex, 1);
}

int main(int argc, char* argv[])
{
	std::map<std::string, solutionFunction> solutions;

	//cpu
	//solutions.insert(std::make_pair("CPU", CPUSolution));

	solutions.insert(std::make_pair("n_squared", NSquaredSolution));

	solutions.insert(std::make_pair("serial_atomic_one_phase", serialAtomicOnePhaseSolution));
	solutions.insert(std::make_pair("serial_atomic", serialAtomicSolution));
	solutions.insert(std::make_pair("serial_atomic_add", serialAtomicADDSolution));
	solutions.insert(std::make_pair("serial_atomic_dd", serialAtomicDDSolution));
	solutions.insert(std::make_pair("serial_scan", serialScanSolution));
	solutions.insert(std::make_pair("serial_scan_dd", serialScanDDSolution));
	solutions.insert(std::make_pair("serial_scan_add", serialScanADDSolution));
	solutions.insert(std::make_pair("serial_half_scan", serialHalfScanSolution));
	solutions.insert(std::make_pair("serial_half_scan_dd", serialHalfScanDDSolution));
	solutions.insert(std::make_pair("serial_half_scan_add", serialHalfScanADDSolution));

	solutions.insert(std::make_pair("warp_atomic_one_phase", warpBasedAtomicSolution));
	solutions.insert(std::make_pair("warp_atomic", warpBasedAtomicSolution));
	solutions.insert(std::make_pair("warp_atomic_add", warpBasedAtomicADDSolution));
	solutions.insert(std::make_pair("warp_atomic_dd", warpBasedAtomicDDSolution));
	solutions.insert(std::make_pair("warp_scan", warpBasedScanSolution));
	solutions.insert(std::make_pair("warp_scan_dd", warpBasedScanDDSolution));
	solutions.insert(std::make_pair("warp_scan_add", warpBasedScanADDSolution));
	solutions.insert(std::make_pair("warp_half_scan", warpBasedHalfScanSolution));
	solutions.insert(std::make_pair("warp_half_scan_dd", warpBasedHalfScanDDSolution));
	solutions.insert(std::make_pair("warp_half_scan_add", warpBasedHalfScanADDSolution));

	solutions.insert(std::make_pair("CTA", CTATwoPhaseSolution));

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

	if (assemble_csr_matrix(argv[1], &rAdjacencyList, &cAdjacencyList, &noVertices, &noEdges) == -1)
	{
		showUsage(solutions);
		return 0;
	}
	printf("Vertices: %d\n", noVertices);
	printf("Edges: %d\n", noEdges);
	int noTests = strtol(argv[2], NULL, 10);
	printf("Number of tests: %d\n", noTests);
	bool cpuSolutionTesting = false;
	float cpuTime = 0;
	if (strcmp(argv[3],"--all") == 0)
	{
		for each (auto var in solutions)
		{
			chosenSolutions.push_back(var.first);
		}
		cpuSolutionTesting = true;
	}
	else
	{
		for (int i = 3; i < argc; i++)
		{
			if (strcmp(argv[i], "CPU") == 0)
			{
				cpuSolutionTesting = true;
			}
			else
			{
				auto sol = solutions.find(argv[i]);
				if (sol == solutions.end())
				{
					printf("Solution %s not found \n", argv[i]);
					free(cAdjacencyList);
					free(rAdjacencyList);
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
		index = 0;
	}
	progressBar(99, 100);
	printf("\n");
	if (cpuSolutionTesting)
	{
		printf("Testing CPU\n");
		cpuTime += launchCPUSolutions(CPUSolution, cAdjacencyList, rAdjacencyList, noVertices, noEdges, 0, noTests);
	}
	printf("\nResults\n");
	for each (std::string var in chosenSolutions)
	{
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


float launchSolution(void (*solution)(int*, int*, int, int, int), int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	solution(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	return milliseconds;
}

float launchCPUSolutions(void(*solution)(int*, int*, int, int, int), int* cAdjacencyList, int* rAdjacencyList, int noVertices, int noEdges, int startingVertex, int noTimes)
{
	 clock_t start, end;
	 start = clock();

	 for(int i=0; i<noTimes; ++i)
		solution(cAdjacencyList, rAdjacencyList, noVertices, noEdges, startingVertex);

	 end = clock();
	 return (float(1000 * (end - start))) / CLOCKS_PER_SEC;
}

void showUsage(std::map<std::string, solutionFunction> solutions)
{
	printf("usage:\t mybfs\t file_mtx\t number_of_tests\t solution1\t [solution2 ...]\n");
	printf("  file_mtx is a path to the graph file in the Matrix Market format\n");
	printf("  available solutions:\n");
	for each (std::pair< std::string, solutionFunction> var in solutions)
	{
		printf("    %s\n", var.first.c_str());
	}
}



